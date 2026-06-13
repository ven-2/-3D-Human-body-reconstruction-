//
//  FrameUploader.swift
//  ProjectCloudExample
//
//  Created by Venkata Ashokkumar on 11/02/2026.
//

import Foundation
import ARKit
import CoreImage
import CoreVideo

actor FrameUploader {
    let baseURL: URL
    let sessionID: String

    private var frameIndex: Int = 0
    private var lastSendTime: TimeInterval = 0
    private let maxFPS: Double = 5.0
    private let ciContext = CIContext()

    init(baseURL: URL, sessionID: String) {
        self.baseURL = baseURL
        self.sessionID = sessionID
    }

    func sendFrame(_ frame: ARFrame) async {
        // throttle
        let now = Date().timeIntervalSince1970
        let minDt = 1.0 / maxFPS
        if now - lastSendTime < minDt { return }
        lastSendTime = now

        guard let depth = (frame.smoothedSceneDepth ?? frame.sceneDepth) else { return }
        guard let depthData = frame.smoothedSceneDepth ?? frame.sceneDepth else { return }
        let depthMap = depthData.depthMap  // CVPixelBuffer (DepthFloat32)

        if let seg = frame.segmentationBuffer {
            if let segResized = resizeSegmentationMaskToDepth(seg, depth: depthMap) {
                applyPersonMaskToDepth(depth: depthMap, mask: segResized, threshold: 128)
            }
        }

        let confMapOpt = depth.confidenceMap

        let depthBytes = copyPlane0Bytes(depthMap)
        let confBytes = confMapOpt.map(copyPlane0Bytes)

        let T_wc = frame.camera.transform
        let K = frame.camera.intrinsics
        let imgRes = frame.camera.imageResolution

        let meta: [String: Any] = [
            "timestamp": frame.timestamp,

            "depth_width": CVPixelBufferGetWidthOfPlane(depthMap, 0),
            "depth_height": CVPixelBufferGetHeightOfPlane(depthMap, 0),
            "depth_bytes_per_row": CVPixelBufferGetBytesPerRowOfPlane(depthMap, 0),

            "conf_present": confMapOpt != nil,
            "conf_width": confMapOpt.map { CVPixelBufferGetWidthOfPlane($0, 0) } ?? 0,
            "conf_height": confMapOpt.map { CVPixelBufferGetHeightOfPlane($0, 0) } ?? 0,
            "conf_bytes_per_row": confMapOpt.map { CVPixelBufferGetBytesPerRowOfPlane($0, 0) } ?? 0,

            "intrinsics": [
              [K[0,0], K[0,1], K[0,2]],
              [K[1,0], K[1,1], K[1,2]],
              [K[2,0], K[2,1], K[2,2]]
            ],


            // camera to world
            "transform": [
              [T_wc[0,0], T_wc[0,1], T_wc[0,2], T_wc[0,3]],
              [T_wc[1,0], T_wc[1,1], T_wc[1,2], T_wc[1,3]],
              [T_wc[2,0], T_wc[2,1], T_wc[2,2], T_wc[2,3]],
              [T_wc[3,0], T_wc[3,1], T_wc[3,2], T_wc[3,3]]
            ],

            "image_resolution": ["w": imgRes.width, "h": imgRes.height],
            "orientation": "portrait"
        ]

        guard let metaJSON = try? JSONSerialization.data(withJSONObject: meta) else { return }

        let idx = frameIndex
        frameIndex += 1

        do {
            try await postMultipartFrame(
                frameIndex: idx,
                metaJSON: metaJSON,
                depthBytes: depthBytes,
                confBytes: confBytes
            )
        } catch {
            print("Upload frame failed:", error)
        }
    }

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

        // Scale to depth size
        let sx = CGFloat(dw) / input.extent.width
        let sy = CGFloat(dh) / input.extent.height
        let resized = input.transformed(by: CGAffineTransform(scaleX: sx, y: sy))

        ciContext.render(resized, to: outPB)
        return outPB
    }
    
    func applyPersonMaskToDepth(depth: CVPixelBuffer, mask: CVPixelBuffer, threshold: UInt8 = 128) {
        CVPixelBufferLockBaseAddress(depth, .readOnly)  // depth input read/write? use [] if writing
        CVPixelBufferLockBaseAddress(mask, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(mask, .readOnly)
            CVPixelBufferUnlockBaseAddress(depth, .readOnly)
        }

        let w = CVPixelBufferGetWidth(depth)
        let h = CVPixelBufferGetHeight(depth)

        guard CVPixelBufferGetWidth(mask) == w, CVPixelBufferGetHeight(mask) == h else { return }

        let depthRowBytes = CVPixelBufferGetBytesPerRow(depth)
        let maskRowBytes  = CVPixelBufferGetBytesPerRow(mask)

        guard let depthBase = CVPixelBufferGetBaseAddress(depth),
              let maskBase  = CVPixelBufferGetBaseAddress(mask) else { return }

        for y in 0..<h {
            let depthRow = depthBase.advanced(by: y * depthRowBytes).assumingMemoryBound(to: Float32.self)
            let maskRow  = maskBase.advanced(by: y * maskRowBytes).assumingMemoryBound(to: UInt8.self)

            for x in 0..<w {
                if maskRow[x] < threshold {
                    depthRow[x] = 0.0  // 0 means “invalid depth” for backend
                }
            }
        }
    }
    

    func stopAndProcess() async {
        let url = baseURL.appendingPathComponent("stop")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"

        let boundary = "Boundary-\(UUID().uuidString)"
        req.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        func addField(_ name: String, _ value: String) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".data(using: .utf8)!)
            body.append("\(value)\r\n".data(using: .utf8)!)
        }

        addField("session_id", sessionID)
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        req.httpBody = body

        _ = try? await URLSession.shared.data(for: req)
    }

    func waitForResult(pollIntervalSeconds: Double = 1.0, timeoutSeconds: Double = 60) async -> URL? {
        let start = Date()
        while Date().timeIntervalSince(start) < timeoutSeconds {
            if await isReady() {
                return await downloadResult()
            }
            try? await Task.sleep(nanoseconds: UInt64(pollIntervalSeconds * 1_000_000_000))
        }
        print("Timed out waiting for result.")
        return nil
    }

    private func isReady() async -> Bool {
        let url = baseURL.appendingPathComponent("status")
        var comps = URLComponents(url: url, resolvingAgainstBaseURL: false)!
        comps.queryItems = [URLQueryItem(name: "session_id", value: sessionID)]
        guard let finalURL = comps.url else { return false }

        do {
            let (data, resp) = try await URLSession.shared.data(from: finalURL)
            guard (resp as? HTTPURLResponse)?.statusCode == 200 else { return false }
            let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            return (obj?["ready"] as? Bool) == true
        } catch {
            return false
        }
    }
    // MARK: - Upload pose JSON (landmarks)
    // Backend will save it under: sessions/<session_id>/pose.json
    func uploadPoseJSON(from fileURL: URL) async {
        let url = baseURL.appendingPathComponent("pose")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"

        // Read file (if it fails, still return gracefully)
        guard let data = try? Data(contentsOf: fileURL) else {
            print("[FrameUploader] Could not read pose file at", fileURL.path)
            return
        }

        let boundary = "Boundary-\(UUID().uuidString)"
        req.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()

        func addField(_ name: String, _ value: String) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".data(using: .utf8)!)
            body.append("\(value)\r\n".data(using: .utf8)!)
        }

        func addFile(_ name: String, filename: String, contentType: String, data: Data) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\n".data(using: .utf8)!)
            body.append("Content-Type: \(contentType)\r\n\r\n".data(using: .utf8)!)
            body.append(data)
            body.append("\r\n".data(using: .utf8)!)
        }

        // Required field so backend can store pose file with the right session.
        addField("session_id", sessionID)
        addFile("pose_json", filename: "pose.json", contentType: "application/json", data: data)

        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        req.httpBody = body

        do {
            let (respData, resp) = try await URLSession.shared.data(for: req)
            let code = (resp as? HTTPURLResponse)?.statusCode ?? -1
            if code != 200 {
                print("[FrameUploader] /pose upload failed status=\(code) body=\(String(data: respData, encoding: .utf8) ?? "<non-utf8>")")
            } else {
                print("[FrameUploader] Uploaded pose JSON (\(data.count) bytes)")
            }
        } catch {
            print("[FrameUploader] /pose upload error:", error)
        }
    }
    private func downloadResult() async -> URL? {
        let url = baseURL.appendingPathComponent("result")
        var comps = URLComponents(url: url, resolvingAgainstBaseURL: false)!
        comps.queryItems = [URLQueryItem(name: "session_id", value: sessionID)]
        guard let finalURL = comps.url else { return nil }

        do {
            let (data, resp) = try await URLSession.shared.data(from: finalURL)
            guard (resp as? HTTPURLResponse)?.statusCode == 200 else { return nil }

            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            let outURL = docs.appendingPathComponent("result.ply")
            try data.write(to: outURL, options: .atomic)
            return outURL
        } catch {
            print("Download result failed:", error)
            return nil
        }
    }

    private func postMultipartFrame(
        frameIndex: Int,
        metaJSON: Data,
        depthBytes: Data,
        confBytes: Data?
    ) async throws {
        let url = baseURL.appendingPathComponent("frame")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"

        let boundary = "Boundary-\(UUID().uuidString)"
        req.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()

        func addField(_ name: String, _ value: String) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".data(using: .utf8)!)
            body.append("\(value)\r\n".data(using: .utf8)!)
        }

        func addFile(_ name: String, filename: String, contentType: String, data: Data) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\n".data(using: .utf8)!)
            body.append("Content-Type: \(contentType)\r\n\r\n".data(using: .utf8)!)
            body.append(data)
            body.append("\r\n".data(using: .utf8)!)
        }

        addField("session_id", sessionID)
        addField("frame_index", String(frameIndex))
        addField("meta_json", String(data: metaJSON, encoding: .utf8) ?? "{}")
        addFile("depth_bin", filename: "depth.bin", contentType: "application/octet-stream", data: depthBytes)

        if let confBytes {
            addFile("conf_bin", filename: "conf.bin", contentType: "application/octet-stream", data: confBytes)
        }

        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        req.httpBody = body

        _ = try await URLSession.shared.data(for: req)
    }
}
