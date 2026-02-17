use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("kernel.ptx");

    println!("cargo:warning=Compiling CUDA kernel...");

    // compute_86にしとけば、とりあえずAmpere以降の全てのGPUで動作
    let mut cmd = Command::new("nvcc");
    cmd.arg("--ptx")
        .arg("--use_fast_math")
        .arg("-arch=compute_86")
        .arg("-code=compute_86");

    #[cfg(target_os = "windows")]
    {
        // Windows/MSVC環境では、nvccにホストコンパイラのパスを明示的に指定する必要がある場合がある
        let msvc_path = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.43.34808/bin/Hostx64/x64";
        cmd.arg("-ccbin").arg(msvc_path);
    }

    let status = cmd
        .arg("src/kernel.cu")
        .arg("-o")
        .arg(&ptx_path)
        .status()
        .expect("nvccの実行に失敗しました。CUDA Toolkitがインストールされ、PATHが通っているか確認してください。");

    if !status.success() {
        panic!("nvccがエラーコードで終了しました: {}", status);
    }

    println!("cargo:rerun-if-changed=src/kernel.cu");
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(target_os = "windows")]
    {
        if let Ok(cuda_path) = env::var("CUDA_HOME") {
            println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
        } else if let Ok(cuda_path) = env::var("CUDA_PATH") {
            println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
        } else {
            // 一般的なパスを探索
            let paths = vec![
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/lib/x64",
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/lib/x64",
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64",
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64",
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64",
            ];
            for path in paths {
                println!("cargo:rustc-link-search=native={}", path);
            }
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        if let Ok(cuda_path) = env::var("CUDA_HOME") {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        } else if let Ok(cuda_path) = env::var("CUDA_PATH") {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        } else {
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        }
    }

    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
