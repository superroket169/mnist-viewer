use filuplex::context::{Context, GpuPref};
use filuplex::ops::{BuiltInShader, BuiltInShaderType, Operation};
use raylib::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

const INPUT_SIZE: usize = 784;
const HIDDEN_SIZE: usize = 512;
const OUTPUT_SIZE: usize = 10;
const WEIGHTS_FILE: &str = "weights.bin";

// Layout
const CELL: i32 = 16;
const GRID_W: i32 = 28 * CELL;
const GRID_H: i32 = 28 * CELL;
const BAR_AREA_H: i32 = 160;
const BRUSH_RADIUS: f32 = 1.4;
const WIN_W: i32 = GRID_W + 200;
const WIN_H: i32 = BAR_AREA_H + GRID_H + 40;
const GRID_X: i32 = 0;
const GRID_Y: i32 = BAR_AREA_H;

#[derive(Serialize, Deserialize)]
struct Weights {
    input_w: Vec<f32>,
    hidden_w: Vec<f32>,
    iteration: usize,
}

fn load_weights() -> Result<Weights, String> {
    let data = std::fs::read(WEIGHTS_FILE)
        .map_err(|e| format!("Cannot read '{}': {}", WEIGHTS_FILE, e))?;
    bincode::deserialize(&data).map_err(|e| format!("Corrupt weights file: {}", e))
}

fn softmax(v: &[f32]) -> Vec<f32> {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|x| x / sum).collect()
}

fn infer(
    pixels: &[f32; INPUT_SIZE],
    input_w: &[f32],
    hidden_w: &[f32],
    op_matmul: &Operation,
    op_sig: &Operation,
) -> Vec<f32> {
    let l1 = op_matmul.run_matmul(pixels, input_w, 1, INPUT_SIZE as u32, HIDDEN_SIZE as u32);
    let l1_s = op_sig.run_fn(&l1);
    let l2 = op_matmul.run_matmul(&l1_s, hidden_w, 1, HIDDEN_SIZE as u32, OUTPUT_SIZE as u32);
    softmax(&l2)
}

fn center_and_scale(pixels: &mut [f32; INPUT_SIZE]) {
    let mut min_x = 28;
    let mut max_x = -1;
    let mut min_y = 28;
    let mut max_y = -1;

    for y in 0..28 {
        for x in 0..28 {
            if pixels[y * 28 + x] > 0.01 {
                if (x as i32) < min_x {
                    min_x = x as i32;
                }
                if (x as i32) > max_x {
                    max_x = x as i32;
                }
                if (y as i32) < min_y {
                    min_y = y as i32;
                }
                if (y as i32) > max_y {
                    max_y = y as i32;
                }
            }
        }
    }

    if min_x > max_x {
        return;
    }

    let w = (max_x - min_x + 1) as f32;
    let h = (max_y - min_y + 1) as f32;
    let scale = 20.0 / w.max(h);

    let mut scaled = [0.0f32; 784];
    let scaled_w = w * scale;
    let scaled_h = h * scale;
    let offset_x = (28.0 - scaled_w) / 2.0;
    let offset_y = (28.0 - scaled_h) / 2.0;

    for y in 0..28 {
        for x in 0..28 {
            let src_x = (x as f32 - offset_x) / scale + min_x as f32;
            let src_y = (y as f32 - offset_y) / scale + min_y as f32;

            if src_x >= 0.0 && src_x <= 27.0 && src_y >= 0.0 && src_y <= 27.0 {
                let x0 = src_x.floor() as usize;
                let x1 = (x0 + 1).min(27);
                let y0 = src_y.floor() as usize;
                let y1 = (y0 + 1).min(27);

                let dx = src_x - x0 as f32;
                let dy = src_y - y0 as f32;

                let v00 = pixels[y0 * 28 + x0];
                let v10 = pixels[y0 * 28 + x1];
                let v01 = pixels[y1 * 28 + x0];
                let v11 = pixels[y1 * 28 + x1];

                let val = v00 * (1.0 - dx) * (1.0 - dy)
                    + v10 * dx * (1.0 - dy)
                    + v01 * (1.0 - dx) * dy
                    + v11 * dx * dy;
                scaled[y * 28 + x] = val.min(1.0);
            }
        }
    }

    let mut cm_x = 0.0;
    let mut cm_y = 0.0;
    let mut total_mass = 0.0;
    for y in 0..28 {
        for x in 0..28 {
            let v = scaled[y * 28 + x];
            cm_x += x as f32 * v;
            cm_y += y as f32 * v;
            total_mass += v;
        }
    }

    if total_mass == 0.0 {
        return;
    }
    cm_x /= total_mass;
    cm_y /= total_mass;

    let shift_x = 13.5 - cm_x;
    let shift_y = 13.5 - cm_y;

    let mut final_pixels = [0.0f32; 784];
    for y in 0..28 {
        for x in 0..28 {
            let src_x = x as f32 - shift_x;
            let src_y = y as f32 - shift_y;

            if src_x >= 0.0 && src_x <= 27.0 && src_y >= 0.0 && src_y <= 27.0 {
                let x0 = src_x.floor() as usize;
                let x1 = (x0 + 1).min(27);
                let y0 = src_y.floor() as usize;
                let y1 = (y0 + 1).min(27);

                let dx = src_x - x0 as f32;
                let dy = src_y - y0 as f32;

                let v00 = scaled[y0 * 28 + x0];
                let v10 = scaled[y0 * 28 + x1];
                let v01 = scaled[y1 * 28 + x0];
                let v11 = scaled[y1 * 28 + x1];

                let val = v00 * (1.0 - dx) * (1.0 - dy)
                    + v10 * dx * (1.0 - dy)
                    + v01 * (1.0 - dx) * dy
                    + v11 * dx * dy;
                final_pixels[y * 28 + x] = val.min(1.0);
            }
        }
    }

    *pixels = final_pixels;
}

fn main() {
    let weights = match load_weights() {
        Ok(w) => {
            println!("Loaded '{}' (iteration {})", WEIGHTS_FILE, w.iteration);
            w
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    let (mut rl, thread) = raylib::init()
        .size(WIN_W, WIN_H)
        .title("MNIST Visualizer")
        .build();

    rl.set_target_fps(60);

    let ctx = Arc::new(Context::new(GpuPref::Default));
    let op_matmul = Operation::new(
        ctx.clone(),
        BuiltInShader::new(BuiltInShaderType::MatrisMul).load(&ctx),
    );
    let op_sig = Operation::new(
        ctx.clone(),
        BuiltInShader::new(BuiltInShaderType::Sigmoid).load(&ctx),
    );

    let input_w = weights.input_w;
    let hidden_w = weights.hidden_w;

    let mut pixels = [0.0f32; INPUT_SIZE];
    let mut probs = vec![0.1f32; OUTPUT_SIZE];

    let mut needs_infer = true;
    while !rl.window_should_close() {
        let mp = rl.get_mouse_position();
        // println!("Mouse: {:?}, FPS: {}", mp, rl.get_fps());

        let panel_x = GRID_W + 10;
        let btn_rect = Rectangle::new(panel_x as f32, (WIN_H - 50) as f32, 160.0, 35.0);
        let mut btn_color = Color::new(50, 50, 70, 255);

        if btn_rect.check_collision_point_rec(mp) {
            btn_color = Color::new(70, 70, 90, 255);
            if rl.is_mouse_button_pressed(MouseButton::MOUSE_BUTTON_LEFT) {
                center_and_scale(&mut pixels);
                needs_infer = true;
            }
        }

        // if rl.is_mouse_button_down(MouseButton::MOUSE_BUTTON_LEFT) {
        //     print!("hey"); }

        if rl.is_mouse_button_down(MouseButton::MOUSE_BUTTON_LEFT)
            && !btn_rect.check_collision_point_rec(mp)
        {
            let mx = (mp.x as i32 - GRID_X) as f32 / CELL as f32;
            let my = (mp.y as i32 - GRID_Y) as f32 / CELL as f32;
            if mx >= 0.0 && mx < 28.0 && my >= 0.0 && my < 28.0 {
                let cx = mx as i32;
                let cy = my as i32;
                for dy in -2..=2i32 {
                    for dx in -2..=2i32 {
                        let px = cx + dx;
                        let py = cy + dy;
                        if px >= 0 && px < 28 && py >= 0 && py < 28 {
                            let dist = ((dx as f32 - (mx - cx as f32)).powi(2)
                                + (dy as f32 - (my - cy as f32)).powi(2))
                            .sqrt();
                            let strength = (1.0 - dist / BRUSH_RADIUS).max(0.0);
                            let idx = (py * 28 + px) as usize;
                            pixels[idx] = (pixels[idx] + strength * 0.4).min(1.0);
                        }
                    }
                }
            }
            needs_infer = true;
        }

        if rl.is_mouse_button_pressed(MouseButton::MOUSE_BUTTON_RIGHT) {
            pixels = [0.0f32; INPUT_SIZE];
            needs_infer = true;
        }

        if needs_infer {
            let mut infer_pixels = pixels.clone();
            center_and_scale(&mut infer_pixels);
            probs = infer(&infer_pixels, &input_w, &hidden_w, &op_matmul, &op_sig);
            needs_infer = false;
        }

        // -----------------------------

        let pred = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::new(18, 18, 24, 255));

        let bar_w = (GRID_W / OUTPUT_SIZE as i32) - 4;
        let bar_max_h = BAR_AREA_H - 50;
        let by_bottom = BAR_AREA_H - 30;

        for i in 0..OUTPUT_SIZE {
            let p = probs[i];
            let bar_h = (p * bar_max_h as f32) as i32;
            let bx = GRID_X + i as i32 * (bar_w + 4) + 2;
            let is_pred = i == pred;

            let bar_color = if is_pred {
                Color::new(80, 200, 120, 255)
            } else {
                Color::new(60, 100, 180, 255)
            };
            let border_color = if is_pred {
                Color::new(80, 220, 140, 255)
            } else {
                Color::new(60, 60, 90, 255)
            };
            let label_color = if is_pred {
                Color::WHITE
            } else {
                Color::new(160, 160, 180, 255)
            };

            d.draw_rectangle(
                bx,
                by_bottom - bar_max_h,
                bar_w,
                bar_max_h,
                Color::new(35, 35, 50, 255),
            );
            d.draw_rectangle(bx, by_bottom - bar_h, bar_w, bar_h, bar_color);
            d.draw_rectangle_lines(bx, by_bottom - bar_max_h, bar_w, bar_max_h, border_color);

            d.draw_text(
                &i.to_string(),
                bx + bar_w / 2 - 5,
                by_bottom + 4,
                18,
                label_color,
            );

            let pct_str = format!("{:.0}%", p * 100.0);
            let pct_x = bx + bar_w / 2 - (pct_str.len() as i32 * 4);
            let pct_y = (by_bottom - bar_h - 18).max(2);
            d.draw_text(&pct_str, pct_x, pct_y, 12, Color::new(200, 200, 220, 255));
        }

        for row in 0..28i32 {
            for col in 0..28i32 {
                let val = pixels[(row * 28 + col) as usize];
                let b = (val * 255.0) as u8;
                d.draw_rectangle(
                    GRID_X + col * CELL,
                    GRID_Y + row * CELL,
                    CELL,
                    CELL,
                    Color::new(b, b, b, 255),
                );
            }
        }

        d.draw_rectangle_lines(GRID_X, GRID_Y, GRID_W, GRID_H, Color::new(80, 80, 110, 255));

        for i in 0..=7i32 {
            d.draw_line(
                GRID_X + i * 4 * CELL,
                GRID_Y,
                GRID_X + i * 4 * CELL,
                GRID_Y + GRID_H,
                Color::new(40, 40, 60, 255),
            );
            d.draw_line(
                GRID_X,
                GRID_Y + i * 4 * CELL,
                GRID_X + GRID_W,
                GRID_Y + i * 4 * CELL,
                Color::new(40, 40, 60, 255),
            );
        }

        let panel_y = BAR_AREA_H;
        d.draw_text(
            "PRED",
            panel_x,
            panel_y + 20,
            16,
            Color::new(140, 140, 160, 255),
        );
        d.draw_text(
            &pred.to_string(),
            panel_x + 10,
            panel_y + 50,
            120,
            Color::new(80, 200, 120, 255),
        );
        d.draw_text(
            &format!("{:.1}%", probs[pred] * 100.0),
            panel_x + 5,
            panel_y + 180,
            22,
            Color::new(180, 220, 180, 255),
        );

        d.draw_text(
            "LMB: Draw",
            panel_x,
            WIN_H - 100,
            14,
            Color::new(100, 100, 120, 255),
        );
        d.draw_text(
            "RMB: Clear",
            panel_x,
            WIN_H - 80,
            14,
            Color::new(100, 100, 120, 255),
        );

        d.draw_rectangle_rec(btn_rect, btn_color);
        d.draw_rectangle_lines_ex(btn_rect, 2.0, Color::new(80, 80, 110, 255));
        d.draw_text("CENTER (C)", panel_x + 35, WIN_H - 40, 16, Color::WHITE);
    }
}
