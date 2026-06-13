//
//  ARManager.swift
//  ProjectCloudExample
//
//  Created by Venkata Ashokkumar on 18/01/2026.
//

import Foundation
import ARKit
import SceneKit
import Observation
import ImageIO
import UIKit
import simd

@MainActor
@Observable
final class ARManager: NSObject, ARSessionDelegate {

    let sceneView    = ARSCNView(frame: .zero)
    let geometryNode = SCNNode()

    var isCapturing   = false
    var resultFileURL: URL? = nil

    private var isProcessing        = false
    private var overlayFrameCounter = 0

    let pointCloud = PointCloud()

    // MARK: - 3D joint sphere overlay

    /// Root node that holds all joint spheres. Lives in world space.
    private let poseRootNode = SCNNode()
    /// Sphere node per joint name — reused across frames so they just move in place.
    private var jointNodes: [String: SCNNode] = [:]

    // MARK: - Pose tracker

    private let poseTracker: PoseTracker

    // MARK: - Backend

    // Point this at your backend server's LAN IP, e.g. "http://192.168.x.x:8000"
    private let backendBaseURL = URL(string: "http://localhost:8000")!
    private let sessionID      = UUID().uuidString
    private let uploader:       FrameUploader

    // MARK: - Init

    override init() {
        self.uploader    = FrameUploader(baseURL: backendBaseURL, sessionID: sessionID)
        self.poseTracker = PoseTracker(
            sessionId: sessionID,
            config: .init(
                runEveryNFrames:    3,
                minJointConfidence: 0.30,
                maxStoredFrames:    2_000,
                visionOrientation:  .right   // rear camera portrait; flip to .left if mirrored
            )
        )
        super.init()

        sceneView.scene = SCNScene()
        sceneView.session.delegate = self
        sceneView.scene.rootNode.addChildNode(geometryNode)
        sceneView.scene.rootNode.addChildNode(poseRootNode)

        let configuration = ARWorldTrackingConfiguration()
        var semantics: ARConfiguration.FrameSemantics = []
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            semantics.insert(.sceneDepth)
        }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.personSegmentationWithDepth) {
            semantics.insert(.personSegmentationWithDepth)
        } else if ARWorldTrackingConfiguration.supportsFrameSemantics(.personSegmentation) {
            semantics.insert(.personSegmentation)
        }
        configuration.frameSemantics = semantics
        sceneView.session.run(configuration)
    }

    // MARK: - Capture control

    func toggleCapture() {
        isCapturing.toggle()

        if isCapturing {
            resultFileURL        = nil
            overlayFrameCounter  = 0
            Task { await pointCloud.reset() }
            Task { await poseTracker.reset() }
            removePoseSpheres()
        } else {
            // On stop: export + upload pose JSON, then kick off TSDF mesh extraction.
            Task {
                let docs    = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                let poseURL = docs.appendingPathComponent("pose_\(sessionID).json")
                _ = await poseTracker.writeSessionJSON(to: poseURL)
                await uploader.uploadPoseJSON(from: poseURL)
                await uploader.stopAndProcess()
                let url = await uploader.waitForResult()
                self.resultFileURL = url
                print("Result PLY:", url as Any)
            }
        }
    }

    // MARK: - ARSessionDelegate

    nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
        Task { @MainActor in await process(frame: frame) }
    }

    private func process(frame: ARFrame) async {
        guard isCapturing, !isProcessing else { return }
        isProcessing = true
        defer { isProcessing = false }

        await poseTracker.process(frame: frame, frameIndex: overlayFrameCounter)
        await pointCloud.process(frame: frame)
        overlayFrameCounter += 1

        // Refresh joint spheres every 3 frames — smooth enough without thrashing SceneKit.
        if overlayFrameCounter % 3 == 0 {
            let joints = await poseTracker.latestJoints3D()
            updatePoseSpheres(joints: joints)
        }

        if overlayFrameCounter % 5 == 0 {
            await updateGeometry()
        }

        await uploader.sendFrame(frame)
    }

    // MARK: - Joint sphere management

    /// Creates or updates one sphere per joint at its world-space position.
    /// Because the node position is in ARKit world space, the sphere stays locked
    /// to the person's body even as you walk around them.
    private func updatePoseSpheres(joints: [PoseTracker.Joint3D]) {
        for j in joints {
            let node: SCNNode

            if let existing = jointNodes[j.name] {
                node = existing
            } else {
                // Build a small cyan sphere once per joint name.
                let sphere        = SCNSphere(radius: 0.025)   // 2.5 cm — visible but not huge
                sphere.segmentCount = 8                         // low-poly for performance

                let mat                  = SCNMaterial()
                mat.lightingModel        = .constant            // unaffected by scene lighting
                mat.diffuse.contents     = UIColor.systemCyan
                mat.transparency         = 0.15                 // slightly see-through
                sphere.materials         = [mat]

                node = SCNNode(geometry: sphere)
                jointNodes[j.name] = node
                poseRootNode.addChildNode(node)
            }

            // SCNNode.position is in the parent node's coordinate space.
            // poseRootNode is a child of scene.rootNode (world space), so this is world metres.
            node.position = SCNVector3(j.x, j.y, j.z)
        }
    }

    private func removePoseSpheres() {
        for (_, node) in jointNodes { node.removeFromParentNode() }
        jointNodes.removeAll()
    }

    // MARK: - Point cloud geometry

    private func updateGeometry() async {
        let verts = await pointCloud.snapshotEveryNth(10)
        guard !verts.isEmpty else { return }

        let positions   = verts.map { SCNVector3($0.position.x, $0.position.y, $0.position.z) }
        let vertexSource = SCNGeometrySource(vertices: positions)

        var colors      = verts.map { $0.color }
        let colorData   = Data(bytes: &colors, count: MemoryLayout<simd_float4>.stride * colors.count)
        let colorSource  = SCNGeometrySource(
            data:               colorData,
            semantic:           .color,
            vectorCount:        verts.count,
            usesFloatComponents: true,
            componentsPerVector: 4,
            bytesPerComponent:  MemoryLayout<Float>.size,
            dataOffset:         0,
            dataStride:         MemoryLayout<simd_float4>.stride
        )

        let indices  = Array(0..<UInt32(verts.count))
        let element  = SCNGeometryElement(indices: indices, primitiveType: .point)
        element.maximumPointScreenSpaceRadius = 15

        let geometry = SCNGeometry(sources: [vertexSource, colorSource], elements: [element])
        geometry.firstMaterial?.lightingModel = .constant
        geometry.firstMaterial?.isDoubleSided = true
        geometryNode.geometry = geometry
    }
}
