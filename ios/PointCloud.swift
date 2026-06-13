//
//  PointCloudProcessor.swift
//  ProjectCloudExample
//
//  Created by Venkata Ashokkumar on 18/01/2026.
//

import Foundation
import ARKit
import simd
import UIKit
import CoreImage


private let _ciContext = CIContext()

func resizeSegmentationMaskToDepth(_ seg: CVPixelBuffer, depth: CVPixelBuffer) -> CVPixelBuffer? {
    let dw = CVPixelBufferGetWidth(depth)
    let dh = CVPixelBufferGetHeight(depth)

    var out: CVPixelBuffer?
    let attrs: [CFString: Any] = [
        kCVPixelBufferPixelFormatTypeKey: Int(kCVPixelFormatType_OneComponent8),
        kCVPixelBufferWidthKey: dw,
        kCVPixelBufferHeightKey: dh,
        kCVPixelBufferIOSurfacePropertiesKey: [:]
    ]

    CVPixelBufferCreate(kCFAllocatorDefault, dw, dh, kCVPixelFormatType_OneComponent8, attrs as CFDictionary, &out)
    guard let outPB = out else { return nil }

    let input = CIImage(cvPixelBuffer: seg)
    let sx = CGFloat(dw) / input.extent.width
    let sy = CGFloat(dh) / input.extent.height
    let resized = input.transformed(by: CGAffineTransform(scaleX: sx, y: sy))

    _ciContext.render(resized, to: outPB)
    return outPB
}

actor PointCloud {

    // A single colored 3D point in world space
    struct Vertex {
        let position: simd_float3   // (x,y,z) meters in AR world space
        let color: simd_float4      // RGBA 0..1
    }

    // GridKey groups nearby 3D points into the same "cell"
    // so we store only ONE representative point per cell.
    struct GridKey: Hashable {

        // Higher = finer grid (more points kept).
        // Lower = coarser grid (more aggressive downsample).
        static let density: Float = 100  // ~ 1/100 m = 1cm-ish cells

        private let ix: Int
        private let iy: Int
        private let iz: Int
        

        init(_ p: simd_float3) {
            // Convert continuous world coordinates into grid indices
            // by scaling and rounding.
            ix = Int((p.x * Self.density).rounded())
            iy = Int((p.y * Self.density).rounded())
            iz = Int((p.z * Self.density).rounded())
        }
    }

    // Persistent merged point cloud across many frames:
    // key = grid cell, value = one representative point
    private(set) var vertices: [GridKey: Vertex] = [:]
    


    // (Optional) helper to clear the cloud when you stop/start capturing
    func reset() {
        vertices.removeAll()
    }
    
    func snapshotEveryNth(_ n: Int) -> [Vertex] {
        Array(vertices.values.enumerated().compactMap { (idx, v) in // help with world build
            (idx % n == n - 1) ? v : nil
        })
    }
    
    func snapshotVertices() -> [Vertex] {
        Array(vertices.values)
    }


    // PART 2: Merge points from this frame into the persistent dictionary
    func process(frame: ARFrame) async {

        // We compute the vertices for THIS frame on main actor
        // (to avoid Swift 6 actor-isolation issues with CVPixelBuffer access).
        let newVertices: [Vertex] = await MainActor.run {

            guard
                let depth = (frame.smoothedSceneDepth ?? frame.sceneDepth),
                let depthPB = Optional(depth.depthMap),
                let depthBuffer = PixelBuffer<Float32>(pixelBuffer: depthPB),
                let confidenceMap = depth.confidenceMap,
                let confidenceBuffer = PixelBuffer<UInt8>(pixelBuffer: confidenceMap),
                let imageBuffer = YCBCRBuffer(pixelBuffer: frame.capturedImage)
            else { return [] }

            let segPB = frame.segmentationBuffer
            var segBuffer: PixelBuffer<UInt8>? = nil

            if let segPB,
               let resized = resizeSegmentationMaskToDepth(segPB, depth: depthPB) {
                segBuffer = PixelBuffer<UInt8>(pixelBuffer: resized)
            }

            var out: [Vertex] = []
            out.reserveCapacity(10_000)

            let depthH = depthBuffer.size.height
            let depthW = depthBuffer.size.width
            let imageSize = imageBuffer.size.asFloat

            // Orientation correction matrix (portrait for now)
            let rotateToARCamera = makeRotateToARCameraMatrix(orientation: .portrait)

            // Camera transform that maps camera-space -> world-space
            let cameraTransform =
                frame.camera.viewMatrix(for: .portrait).inverse * rotateToARCamera

            // Inverse intrinsics maps pixel -> camera ray
            let intrinsicsInv = simd_inverse(frame.camera.intrinsics)

            // Iterate depth pixels
            for row in 0..<depthH {
                for col in 0..<depthW {
                    // filter human
                    if let segBuffer {
                        let m = segBuffer.value(x: col, y: row) // 0..255
                        if m < 128 { continue }                 // keep only person
                    }
                    // Confidence 1..3 (ARConfidenceLevel)
                    let raw = Int(confidenceBuffer.value(x: col, y: row))
                    guard let conf = ARConfidenceLevel(rawValue: raw),
                          conf == .high
                    else { continue }
                    

                    // Depth in meters
                    let d = depthBuffer.value(x: col, y: row)
                    if d.isNaN || d <= 0 { continue }
                    if d > 2.0 { continue }   // IMPORTANT: continue, not return
                    

                    // Normalized depth pixel coords in [0,1]
                    let n = simd_float2(
                        Float(col) / Float(depthW),
                        Float(row) / Float(depthH)
                    )

                    // Map to image pixel coords + clamp
                    let px = min(max(Int((n.x * imageSize.x).rounded()), 0), Int(imageSize.x - 1))
                    let py = min(max(Int((n.y * imageSize.y).rounded()), 0), Int(imageSize.y - 1))

                    // Sample color from camera image
                    let color = imageBuffer.color(x: px, y: py)

                    // ---- 2D -> 3D conversion ----
                    // Build homogeneous pixel point on image plane
                    let screenPoint = simd_float3(n * imageSize, 1.0)

                    // Convert pixel to 3D camera-space point using intrinsics + depth
                    let localPoint = intrinsicsInv * screenPoint * d

                    // Convert camera-space to world-space
                    let world4 = cameraTransform * simd_float4(localPoint, 1.0)

                    // Normalize homogeneous coordinate
                    let world = simd_float3(world4.x, world4.y, world4.z) / world4.w
                    // -------------------------------

                    out.append(Vertex(position: world, color: color))
                }
            }

            return out
        }

        // Now we are back inside the PointCloud actor (off-main).
        // Merge points into the persistent grid dictionary.
        for v in newVertices {
            let key = GridKey(v.position)

            // Only store the first point that lands in this grid cell.
            // (Tutorial does this; later you could average points instead.)
            if vertices[key] == nil {
                vertices[key] = v
            }
        }

        if !newVertices.isEmpty {
            print("Frame points:", newVertices.count, "Total merged:", vertices.count)
        }
    }
}



