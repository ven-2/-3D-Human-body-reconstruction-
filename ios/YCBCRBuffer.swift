//
//  YCBCRBuffer.swift
//  ProjectCloudExample
//
//  Created by Venkata Ashokkumar on 18/01/2026.
//
import CoreVideo 
import simd  
final class YCBCRBuffer {
    
    let size: Size
    
    private let pixelBuffer: CVPixelBuffer
    private let yPlane: UnsafeMutableRawPointer
    private let cbCrPlane: UnsafeMutableRawPointer
    private let ySize: Size
    private let cbCrSize: Size
    
    init?(pixelBuffer: CVPixelBuffer) {
        self.pixelBuffer = pixelBuffer
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        
        guard let yPlane = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0),
                let cbCrPlane = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1) else {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
            return nil
        }
        
        self.yPlane = yPlane
        self.cbCrPlane = cbCrPlane
 
        size = .init(width: CVPixelBufferGetWidth(pixelBuffer),
                     height: CVPixelBufferGetHeight(pixelBuffer))
        
        ySize = .init(width: CVPixelBufferGetWidthOfPlane(pixelBuffer, 0),
                      height: CVPixelBufferGetHeightOfPlane(pixelBuffer, 0))
        
        cbCrSize = .init(width: CVPixelBufferGetWidthOfPlane(pixelBuffer, 1),
                         height: CVPixelBufferGetHeightOfPlane(pixelBuffer, 1))
    }
    
    func color(x: Int, y: Int) -> simd_float4 {
        let yIndex = y * CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0) + x
        let uvIndex = y / 2 * CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1) + x / 2 * 2
        
        // Extract the Y, Cb, and Cr values
        let yValue = yPlane.advanced(by: yIndex)
                .assumingMemoryBound(to: UInt8.self).pointee

        let cbValue = cbCrPlane.advanced(by: uvIndex)
                .assumingMemoryBound(to: UInt8.self).pointee

        let crValue = cbCrPlane.advanced(by: uvIndex + 1)
                .assumingMemoryBound(to: UInt8.self).pointee
        
        // Convert YCbCr to RGB
        let y = Float(yValue) - 16
        let cb = Float(cbValue) - 128
        let cr = Float(crValue) - 128
        
        let r = 1.164 * y + 1.596 * cr
        let g = 1.164 * y - 0.392 * cb - 0.813 * cr
        let b = 1.164 * y + 2.017 * cb
        
        // normalize rgb components
        return simd_float4(max(0, min(255, r)) / 255.0,
                           max(0, min(255, g)) / 255.0,
                           max(0, min(255, b)) / 255.0, 1.0)
    }
    
    deinit {
        CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
    }
}
