//
//  PoseTracker.swift
//  ProjectCloudExample
//
//  Created by Venkata Ashokkumar on 18/01/2026.
//  3D-only pose tracking.
//  Detects body joints with Vision on each AR frame.
//  Back-projects them into ARKit world space using depth + intrinsics.
//  World-space joints stay anchored in the scene as the camera moves.
//  Accumulates all frames and exports a per-joint average for the backend.


import Foundation
import ARKit
import Vision
import ImageIO
import UIKit
import simd

actor PoseTracker {

    // MARK: - Codable output types

    struct Joint3D: Codable {
        let name: String
        let x: Float
        let y: Float
        let z: Float
        let depthMeters: Float
        let confidence2D: Float
    }

    struct PoseFrame: Codable {
        let frameIndex: Int
        let timestamp: Double
        let imageWidth: Int   // portrait width
        let imageHeight: Int  // portrait height
        let joints3D: [Joint3D]
    }

    struct PoseSession: Codable {
        let sessionId: String
        let createdAt: Double
        let frames: [PoseFrame]
        /// Mean world position per joint — the backend should use this for sizing.
        let averagedJoints3D: [Joint3D]
    }

    // MARK: - Config

    struct Config {
        /// Run Vision every N processed frames to keep CPU sane.
        var runEveryNFrames: Int = 3
        /// Discard joints below this Vision confidence.
        var minJointConfidence: Float = 0.30
        /// Hard cap on stored frames to prevent memory blowup.
        var maxStoredFrames: Int = 2_000
        /// For ARKit rear camera in portrait mode the pixel buffer arrives landscape-rotated.
        /// .right tells Vision to rotate it 90° CCW so it sees a portrait image.
        /// Flip to .left if joints appear mirrored.
        var visionOrientation: CGImagePropertyOrientation = .right
        /// When depth at exact joint pixel is 0/NaN, search this many pixels around it.
        var depthSearchRadius: Int = 5
        var maxDepthMeters: Float = 2.5
        var minDepthMeters: Float = 0.10
    }

    // MARK: - Internal state

    private let sessionId: String
    private var config: Config
    private var frames: [PoseFrame] = []
    private var frameCounter: Int = 0
    /// jointName → all 3D observations across the capture session
    private var accumulated: [String: [Joint3D]] = [:]

    // MARK: - Init

    init(sessionId: String, config: Config = Config()) {
        self.sessionId = sessionId
        self.config    = config
    }

    // MARK: - Public interface

    func reset() {
        frames.removeAll(keepingCapacity: true)
        accumulated.removeAll(keepingCapacity: true)
        frameCounter = 0
    }

    /// Call once per AR frame while capturing.
    func process(frame: ARFrame, frameIndex: Int) async {
        frameCounter += 1
        guard frameCounter % max(1, config.runEveryNFrames) == 0 else { return }
        guard frames.count < config.maxStoredFrames else { return }

        // Snapshot config so the MainActor closure doesn't touch actor state.
        let minConf   = config.minJointConfidence
        let visionOri = config.visionOrientation
        let radius    = config.depthSearchRadius
        let maxD      = config.maxDepthMeters
        let minD      = config.minDepthMeters

        let result: PoseFrame? = await MainActor.run {
            PoseDetector.detect(
                frame:              frame,
                frameIndex:         frameIndex,
                minJointConfidence: minConf,
                visionOrientation:  visionOri,
                depthSearchRadius:  radius,
                minDepthMeters:     minD,
                maxDepthMeters:     maxD
            )
        }

        if let pf = result {
            frames.append(pf)
            for j in pf.joints3D {
                accumulated[j.name, default: []].append(j)
            }
        }
    }

    /// Latest frame's joints — used by ARManager to position SceneKit spheres.
    func latestJoints3D() -> [Joint3D] {
        frames.last?.joints3D ?? []
    }

    /// Mean world position per joint across all captured frames.
    func averagedJoints3D() -> [Joint3D] {
        accumulated.compactMap { name, joints -> Joint3D? in
            guard !joints.isEmpty else { return nil }
            let n = Float(joints.count)
            return Joint3D(
                name:          name,
                x:             joints.map(\.x).reduce(0, +) / n,
                y:             joints.map(\.y).reduce(0, +) / n,
                z:             joints.map(\.z).reduce(0, +) / n,
                depthMeters:   joints.map(\.depthMeters).reduce(0, +) / n,
                confidence2D:  joints.map(\.confidence2D).reduce(0, +) / n
            )
        }
    }

    // MARK: - JSON export

    func exportSessionJSON(pretty: Bool = true) -> Data? {
        let session = PoseSession(
            sessionId:        sessionId,
            createdAt:        Date().timeIntervalSince1970,
            frames:           frames,
            averagedJoints3D: averagedJoints3D()
        )
        let enc = JSONEncoder()
        if pretty { enc.outputFormatting = [.prettyPrinted, .sortedKeys] }
        return try? enc.encode(session)
    }

    func writeSessionJSON(to url: URL) -> Bool {
        guard let data = exportSessionJSON() else { return false }
        do {
            try data.write(to: url, options: .atomic)
            print("[PoseTracker] Wrote \(data.count) bytes →", url.lastPathComponent)
            return true
        } catch {
            print("[PoseTracker] Write error:", error); return false
        }
    }
}

// MARK: - PoseDetector (MainActor, no actor isolation issues)

/// All Vision and depth access lives here, safely on the main thread.
@MainActor
enum PoseDetector {

    // Internal 2D joint scratch type
    private struct J2 {
        let name: String
        let xPx: Int    // portrait top-left coords
        let yPx: Int
        let conf: Float
    }

    static func detect(
        frame:              ARFrame,
        frameIndex:         Int,
        minJointConfidence: Float,
        visionOrientation:  CGImagePropertyOrientation,
        depthSearchRadius:  Int,
        minDepthMeters:     Float,
        maxDepthMeters:     Float
    ) -> PoseTracker.PoseFrame? {

        let pixBuf = frame.capturedImage

        // Pixel buffer is always landscape (bufW > bufH for rear camera).
        let bufW = CVPixelBufferGetWidth(pixBuf)
        let bufH = CVPixelBufferGetHeight(pixBuf)

        // After Vision applies .right orientation (90° CCW rotation) the image is portrait.
        // portraitW = bufH (shorter side), portraitH = bufW (longer side).
        let portraitW = bufH
        let portraitH = bufW

        // ---- Run Vision ----
        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixBuf,
            orientation:   visionOrientation,
            options:       [:]
        )
        let req = VNDetectHumanBodyPoseRequest()
        do { try handler.perform([req]) } catch {
            print("[PoseDetector] Vision error:", error); return nil
        }

        guard let obs = req.results?.first as? VNHumanBodyPoseObservation else { return nil }

        let recognized: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint]
        do { recognized = try obs.recognizedPoints(.all) } catch { return nil }

        // ---- Convert Vision normalised → portrait pixel coords ----
        // Vision normalised: origin bottom-left, [0..1].
        // Portrait pixel:    origin top-left, range [0, portraitW-1] × [0, portraitH-1].
        //   xPx = xNorm × (portraitW-1)
        //   yPx = (1 - yNorm) × (portraitH-1)
        var joints2D: [J2] = []
        for (jname, pt) in recognized {
            let conf = Float(pt.confidence)
            guard conf >= minJointConfidence else { continue }

            let xN = Float(pt.location.x)
            let yN = Float(pt.location.y)

            let xPx = clamp(Int((xN * Float(portraitW - 1)).rounded()), 0, portraitW - 1)
            let yPx = clamp(Int(((1.0 - yN) * Float(portraitH - 1)).rounded()), 0, portraitH - 1)

            joints2D.append(J2(name: jname.rawValue.rawValue, xPx: xPx, yPx: yPx, conf: conf))
        }
        guard !joints2D.isEmpty else { return nil }

        // ---- 3D lifting ----
        let joints3D = liftToWorld(
            joints2D:    joints2D,
            frame:       frame,
            portraitW:   portraitW,
            portraitH:   portraitH,
            bufW:        bufW,
            bufH:        bufH,
            radius:      depthSearchRadius,
            minD:        minDepthMeters,
            maxD:        maxDepthMeters
        )
        guard !joints3D.isEmpty else { return nil }

        return PoseTracker.PoseFrame(
            frameIndex:  frameIndex,
            timestamp:   frame.timestamp,
            imageWidth:  portraitW,
            imageHeight: portraitH,
            joints3D:    joints3D
        )
    }

    // MARK: - Depth helpers

    /// Reads a Float32 depth value. The caller must hold the buffer's base address lock.
    private static func readDepthLocked(_ base: UnsafeRawPointer, bpr: Int, x: Int, y: Int) -> Float32 {
        base.advanced(by: y * bpr).assumingMemoryBound(to: Float32.self)[x]
    }

    /// Searches a (2·radius+1)² window around (cx,cy) for the nearest valid depth.
    private static func nearestDepth(
        _ pb: CVPixelBuffer,
        cx: Int, cy: Int,
        radius: Int,
        minD: Float, maxD: Float
    ) -> Float32? {
        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let bpr = CVPixelBufferGetBytesPerRow(pb)

        var best: Float32? = nil
        var bestDist2 = Int.max

        for dy in -radius...radius {
            let yy = cy + dy; guard yy >= 0, yy < h else { continue }
            for dx in -radius...radius {
                let xx = cx + dx; guard xx >= 0, xx < w else { continue }
                let d = readDepthLocked(base, bpr: bpr, x: xx, y: yy)
                guard d.isFinite, d >= minD, d <= maxD else { continue }
                let dist2 = dx*dx + dy*dy
                if dist2 < bestDist2 { bestDist2 = dist2; best = d }
                if dist2 == 0 { return best }
            }
        }
        return best
    }

    // MARK: - 3D lifting

    /// Back-projects portrait-pixel joints into ARKit world space.
    ///
    /// Key coordinate chain:
    ///   portrait pixel
    ///      landscape buffer pixel   (reverse the .right 90° rotation)
    ///      depth-map pixel          (scale down to depth resolution)
    ///      camera-space 3D point    (K⁻¹ × screenPt × depth)
    ///      world-space 3D point     (T_cw × camPt)
    ///
    /// Portrait pixel  landscape buffer pixel (.right = 90° CCW rotation of buffer):
    ///   bufX = yPx   × (bufW-1) / (portraitH-1)
    ///   bufY = (portraitW-1 - xPx) × (bufH-1) / (portraitW-1)
    private static func liftToWorld(
        joints2D:  [J2],
        frame:     ARFrame,
        portraitW: Int, portraitH: Int,
        bufW:      Int, bufH:      Int,
        radius:    Int,
        minD:      Float, maxD: Float
    ) -> [PoseTracker.Joint3D] {

        guard let depthData = frame.smoothedSceneDepth ?? frame.sceneDepth else { return [] }
        let depthPB = depthData.depthMap
        let depthW  = CVPixelBufferGetWidth(depthPB)
        let depthH  = CVPixelBufferGetHeight(depthPB)

        // Intrinsics are for the landscape buffer.
        let Kinv = simd_inverse(frame.camera.intrinsics)
        // Camera-to-world transform for portrait orientation.
        let rotM = makeRotateToARCameraMatrix(orientation: .portrait)
        let T_cw = frame.camera.viewMatrix(for: .portrait).inverse * rotM

        var out: [PoseTracker.Joint3D] = []
        out.reserveCapacity(joints2D.count)

        for j in joints2D {

            // portrait pixel → landscape buffer pixel
            let bufXf = Float(j.yPx)  * Float(bufW - 1) / Float(max(1, portraitH - 1))
            let bufYf = Float(portraitW - 1 - j.xPx) * Float(bufH - 1) / Float(max(1, portraitW - 1))
            let bufX  = clamp(Int(bufXf.rounded()), 0, bufW - 1)
            let bufY  = clamp(Int(bufYf.rounded()), 0, bufH - 1)

            // landscape buffer pixel → depth-map pixel
            let xD = clamp(Int((Float(bufX) / Float(bufW - 1) * Float(depthW - 1)).rounded()), 0, depthW - 1)
            let yD = clamp(Int((Float(bufY) / Float(bufH - 1) * Float(depthH - 1)).rounded()), 0, depthH - 1)

            // sample depth
            guard let d = nearestDepth(depthPB, cx: xD, cy: yD,
                                       radius: radius, minD: minD, maxD: maxD) else { continue }

            // back-project (intrinsics work in landscape buffer pixel coords)
            let screenPt = simd_float3(bufXf, bufYf, 1.0)
            let camPt    = (Kinv * screenPt) * d

            // camera → world
            let w4    = T_cw * simd_float4(camPt, 1.0)
            let world = simd_float3(w4.x, w4.y, w4.z) / w4.w

            out.append(PoseTracker.Joint3D(
                name:         j.name,
                x:            world.x,
                y:            world.y,
                z:            world.z,
                depthMeters:  d,
                confidence2D: j.conf
            ))
        }
        return out
    }

    private static func clamp(_ v: Int, _ lo: Int, _ hi: Int) -> Int {
        Swift.min(Swift.max(v, lo), hi)
    }
}
