//
//  PixelBufferExtract.swift
//  ProjectCloudExample
//
//  Created by Venkata Ashokkumar on 11/02/2026.
//

import Foundation
import CoreVideo

func copyPlane0Bytes(_ pb: CVPixelBuffer) -> Data {
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }

    guard let base = CVPixelBufferGetBaseAddressOfPlane(pb, 0) else {
        return Data()
    }
    let bpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
    let h = CVPixelBufferGetHeightOfPlane(pb, 0)
    return Data(bytes: base, count: bpr * h)
}

