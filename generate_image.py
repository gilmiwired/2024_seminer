import torch
import os
from torchvision import transforms
from diffusion_model_demo import model_dir, show_images, UNet, Diffuser

# デバイスの設定（GPUが利用可能ならGPUを、そうでなければCPUを使用）
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 生成画像を保存するディレクトリを設定し、存在しない場合は作成
output_dir = 'saved_image'
os.makedirs(output_dir, exist_ok=True)

def generate_and_save_images(model_path, diffuser, device, output_dir, num_images=20):
    """
    指定されたモデルをロードし、画像を生成して保存する関数。

    Args:
        model_path (str): 保存されたモデルのファイルパス。
        diffuser (Diffuser): 拡散モデルのインスタンス。
        device (str): 使用するデバイス（'cuda' または 'cpu'）。
        output_dir (str): 生成画像を保存するディレクトリ。
        num_images (int): 生成する画像の枚数（デフォルトは20）。
    """
    # モデルのインスタンスを新たに作成
    model = UNet(in_ch=1, time_embed_dim=100).to(device)
    
    # モデルの重みをロード
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # モデルを評価モードに設定

    # 画像を生成
    generated_images = diffuser.sample(model, x_shape=(num_images, 1, 28, 28))

    # モデルファイル名からエポック番号を抽出
    epoch_num = os.path.splitext(os.path.basename(model_path))[0].split('_')[-1]

    # 生成された各画像を保存
    for idx, img in enumerate(generated_images):
        # 画像の保存パスを作成（例: epoch_10_img_1.png）
        save_path = os.path.join(output_dir, f'epoch_{epoch_num}_img_{idx+1}.png')
        img.save(save_path)
    print(f'{model_path} から画像を生成し、{output_dir} に保存しました')

    # 生成された画像をテンソルに変換して表示
    tensor_images = torch.stack([transforms.ToTensor()(img) for img in generated_images])
    show_images(tensor_images, rows=2, cols=10)

if __name__ == "__main__":
    # 拡散モデルのインスタンスを作成
    diffuser = Diffuser(num_timesteps=1000, device=device)

    # 特定のディレクトリから保存されたモデルファイルをリストアップ
    saved_model_files = [f for f in os.listdir(model_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]
    # エポック番号でソート（昇順）
    saved_model_files.sort(key=lambda x: int(x.split('_')[-1].split('.pth')[0]))

    # 各保存されたモデルファイルに対して画像を生成・保存
    for model_file in saved_model_files:
        model_path = os.path.join(model_dir, model_file)
        generate_and_save_images(model_path, diffuser, device, output_dir, num_images=20)
