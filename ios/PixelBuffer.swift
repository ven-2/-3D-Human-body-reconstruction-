//
//  PixelBuffer.swift
//  ProjectCloudExample
//
//  Created by Venkata Ashokkumar on 18/01/2026.
//

//struct for storing CVPixelBuffer resolution
import CoreVideo   
import simd  
struct Size {
    let width: Int
    let height: Int
    
    var asFloat: simd_float2 {
        simd_float2(Float(width), Float(height))
    }
}

final class PixelBuffer<T> {
    
    let size: Size
    let bytesPerRow: Int

    private let pixelBuffer: CVPixelBuffer
    private let baseAddress: UnsafeMutableRawPointer
    
    init?(pixelBuffer: CVPixelBuffer) {
        self.pixelBuffer = pixelBuffer

        // lock the buffer while we are getting its values
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        
        guard let baseAddress = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) else {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
            return nil
        }
        self.baseAddress = baseAddress
        
        size = .init(width: CVPixelBufferGetWidth(pixelBuffer),
                     height: CVPixelBufferGetHeight(pixelBuffer))
        bytesPerRow =  CVPixelBufferGetBytesPerRow(pixelBuffer)
    }
    
    // obtain value from pixel buffer in specified coordinates
    func value(x: Int, y: Int) -> T {

        // move to the specified address and get the value bounded to our type
        let rowPtr = baseAddress.advanced(by: y * bytesPerRow)
        return rowPtr.assumingMemoryBound(to: T.self)[x]
    }
    
    deinit {
        CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
    }
}
