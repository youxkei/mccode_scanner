mod scanner;

use image::{DynamicImage, RgbaImage};
use wasm_bindgen::prelude::*;
use wasm_bindgen::Clamped;

#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen(start)]
pub fn main_js() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    Ok(())
}

#[wasm_bindgen]
pub fn scan(width: u32, height: u32, pixels: Clamped<Vec<u8>>) -> Option<String> {
    scanner::scan(
        DynamicImage::ImageRgba8(RgbaImage::from_raw(width, height, pixels.to_vec()).unwrap()),
        false,
    )
    .ok()
}
