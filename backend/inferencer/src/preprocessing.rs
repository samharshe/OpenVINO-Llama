use image::{Rgb, imageops};
use log::info;

/// Convert JPEG bytes to raw BGR tensor in NCHW format
/// Returns a Vec<u8> containing f32 values for a 224x224x3 image
pub fn jpeg_to_raw_bgr(jpeg_bytes: &[u8]) -> Result<Vec<u8>, String> {
    info!("Starting JPEG to BGR conversion");
    
    // Load JPEG from bytes
    let img = image::load_from_memory_with_format(jpeg_bytes, image::ImageFormat::Jpeg)
        .map_err(|e| format!("Failed to decode JPEG: {}", e))?;
    info!("Successfully decoded JPEG image");
    
    // Convert to RGB8 format and resize to 224x224
    let rgb_img = img.to_rgb8();
    let resized = imageops::resize(&rgb_img, 224, 224, imageops::FilterType::Lanczos3);
    info!("Resized image to 224x224");
    
    // Convert to NCHW format (channels first)
    // OpenVINO expects BGR order, so we'll swap R and B
    let mut nchw_data = vec![0f32; 224 * 224 * 3];
    
    for (x, y, pixel) in resized.enumerate_pixels() {
        let idx = (y * 224 + x) as usize;
        let Rgb([r, g, b]) = *pixel;
        
        // Convert to float WITHOUT normalization (keep 0-255 range)
        // Note: BGR order for OpenVINO compatibility
        nchw_data[idx] = b as f32;                    // Blue channel
        nchw_data[224 * 224 + idx] = g as f32;       // Green channel  
        nchw_data[2 * 224 * 224 + idx] = r as f32;   // Red channel
    }
    info!("Converted image to NCHW format with BGR channel order");
    
    // Convert f32 array to bytes
    let bytes = unsafe {
        std::slice::from_raw_parts(
            nchw_data.as_ptr() as *const u8,
            nchw_data.len() * std::mem::size_of::<f32>()
        )
    };
    
    info!("Successfully converted image to raw bytes");
    Ok(bytes.to_vec())
}

/// Check if the data looks like a JPEG image
pub fn is_jpeg(data: &[u8]) -> bool {
    // JPEG magic bytes: FF D8 FF
    data.len() >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF
}

/// Check if the data looks like a preprocessed tensor
pub fn is_tensor(data: &[u8]) -> bool {
    // Tensor should be exactly 224*224*3*4 bytes (4 bytes per f32)
    data.len() == 224 * 224 * 3 * std::mem::size_of::<f32>()
}