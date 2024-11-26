import math
import os
import torch
import torchvision
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim import Adam
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

# ハイパーパラメータの設定
img_size = 28         # 入力画像のサイズ
batch_size = 128      # バッチサイズ
num_timesteps = 1000  # 拡散過程のタイムステップ数
epochs = 30           # エポック数
lr = 1e-3             # 学習率
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # デバイスの設定

model_dir = 'saved_model'
os.makedirs(model_dir, exist_ok=True)

# 画像をグリッド状に表示する関数
def show_images(images, rows=2, cols=10):
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            if i >= len(images):
                break
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(images[i].cpu().numpy().squeeze(), cmap='gray')  # 画像を表示
            ax.axis('off')  # 軸を非表示に
            i += 1
    plt.show()

# 単一の時間インデックスに対する位置エンコーディングを計算する関数
def _pos_encoding(time_idx, output_dim, device='cpu'):
    t, D = time_idx, output_dim
    v = torch.zeros(D, device=device)
    i = torch.arange(0, D, device=device)
    div_term = torch.exp(i / D * math.log(10000))
    v[0::2] = torch.sin(t / div_term[0::2])  # 偶数インデックスにサイン関数を適用
    v[1::2] = torch.cos(t / div_term[1::2])  # 奇数インデックスにコサイン関数を適用
    return v

# バッチ内の各タイムステップに対する位置エンコーディングを計算する関数
def pos_encoding(timesteps, output_dim, device='cpu'):
    batch_size = len(timesteps)
    device = timesteps.device
    v = torch.zeros(batch_size, output_dim, device=device)
    for i in range(batch_size):
        v[i] = _pos_encoding(timesteps[i], output_dim, device)
    return v

# 時間埋め込みを含む畳み込みブロックの定義
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        # 畳み込み層の定義
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),  # 1つ目の畳み込み層
            nn.BatchNorm2d(out_ch),                  # バッチ正規化
            nn.ReLU(),                               # 活性化関数
            nn.Conv2d(out_ch, out_ch, 3, padding=1), # 2つ目の畳み込み層
            nn.BatchNorm2d(out_ch),                  # バッチ正規化
            nn.ReLU()                                # 活性化関数
        )
        # 時間埋め込み用のMLP
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch),        # 時間埋め込みをin_ch次元に変換
            nn.ReLU(),                               # 活性化関数
            nn.Linear(in_ch, in_ch)                  # 再度in_ch次元に変換
        )

    def forward(self, x, v):
        N, C, _, _ = x.shape
        # 時間埋め込みをMLPに通す
        v = self.mlp(v)
        v = v.view(N, C, 1, 1)  # xの形状に合わせてリシェイプ
        # xに時間埋め込みを加えてから畳み込み層に通す
        y = self.convs(x + v)
        return y

# UNetモデルの定義
class UNet(nn.Module):
    def __init__(self, in_ch=1, time_embed_dim=100):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        # ダウンサンプリングパス
        self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        # ボトルネック
        self.bot1 = ConvBlock(128, 256, time_embed_dim)
        # アップサンプリングパス
        self.up2 = ConvBlock(128 + 256, 128, time_embed_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)
        # 出力層
        self.out = nn.Conv2d(64, in_ch, 1)
        # プーリングとアップサンプリング
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, timesteps):
        # タイムステップに対する位置エンコーディングを取得
        v = pos_encoding(timesteps, self.time_embed_dim, x.device)
        # ダウンサンプリングパス
        x1 = self.down1(x, v)   # 最初のダウンサンプリングブロック
        x = self.maxpool(x1)    # マックスプーリング
        x2 = self.down2(x, v)   # 2番目のダウンサンプリングブロック
        x = self.maxpool(x2)    # マックスプーリング
        # ボトルネック
        x = self.bot1(x, v)
        # アップサンプリングパス
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)  # スキップコネクションを結合
        x = self.up2(x, v)             # 最初のアップサンプリングブロック
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)  # スキップコネクションを結合
        x = self.up1(x, v)             # 2番目のアップサンプリングブロック
        # 出力層
        x = self.out(x)
        return x

# 拡散モデル（Diffusion Model）の定義
class Diffuser:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        # β（ノイズスケジュール）の線形スケジュールを定義
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        # αを計算（α = 1 - β）
        self.alphas = 1 - self.betas
        # 累積積 ᾱ を計算（ᾱ_t = ∏_{s=1}^t α_s）
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    # タイムステップtで画像にノイズを追加する関数
    def add_noise(self, x_0, t):
        """
        オリジナルの画像 x_0 に対して、タイムステップ t でノイズを追加した画像 x_t を生成する。
        """
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()
        t_idx = t - 1  # alpha_bars[0] は t=1 に対応
        alpha_bar = self.alpha_bars[t_idx]
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        # 標準正規分布からノイズを生成
        noise = torch.randn_like(x_0, device=self.device)
        # ノイズを追加した画像 x_t を計算
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    # タイムステップtで画像をデノイズする関数（逆拡散過程）
    def denoise(self, model, x, t):
        """
        モデルを使用して、ノイズの入った画像 x_t をタイムステップ t から x_{t-1} にデノイズする。
        """
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()
        t_idx = t - 1
        alpha = self.alphas[t_idx]
        alpha_bar = self.alpha_bars[t_idx]
        # 前のタイムステップの ᾱ を取得
        t_idx_prev = t_idx - 1
        t_idx_prev = t_idx_prev.clamp(min=0)
        alpha_bar_prev = self.alpha_bars[t_idx_prev]
        # t=1 の場合、alpha_bar_prev を 1.0 に設定
        alpha_bar_prev[t_idx == 0] = 1.0

        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        # モデルを使用してノイズ ε を予測
        model.eval()
        with torch.no_grad():
            eps = model(x, t)
        model.train()
        # ランダムなノイズを生成（t=1 の場合はノイズを追加しない）
        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0
        # 逆拡散過程の式に基づいて x_{t-1} を計算
        mu = (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * eps) / torch.sqrt(alpha)
        std = torch.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
        x_prev = mu + noise * std
        return x_prev

    # テンソルをPIL画像に変換する関数
    def reverse_to_img(self, x):
        """
        モデルの出力を画像に変換するための関数。
        """
        x = x.squeeze()
        x = x * 0.5 + 0.5  # [-1,1] から [0,1] にスケーリング
        x = x.clamp(0, 1)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)

    # 逆拡散プロセスを使用してサンプルを生成する関数
    def sample(self, model, x_shape=(20, 1, 28, 28)):
        """
        ノイズから開始し、モデルを使用してクリーンな画像を生成する。
        """
        batch_size = x_shape[0]
        # ランダムなノイズから開始
        x = torch.randn(x_shape, device=self.device)
        # 逆拡散過程をタイムステップ1まで繰り返す
        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t)
        # テンソルを画像に変換
        images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return images

if __name__ == "__main__":
    # Fashion-MNISTデータセットの読み込みと前処理
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 画像を [-1, 1] に正規化
    ])
    dataset = torchvision.datasets.FashionMNIST(root='./data', download=True, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # データセットからサンプル画像を表示
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    show_images(images[:20], rows=2, cols=10)

    # モデルとオプティマイザの定義
    model = UNet(in_ch=1, time_embed_dim=100).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    diffuser = Diffuser(num_timesteps=num_timesteps, device=device)

    # 学習ループ
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for images, _ in tqdm(dataloader):
            images = images.to(device)
            N = images.size(0)
            # 各バッチに対してランダムなタイムステップをサンプリング
            t = torch.randint(1, num_timesteps + 1, (N,), device=device, dtype=torch.long)
            # タイムステップ t で画像にノイズを追加
            x_t, noise = diffuser.add_noise(images, t)
            optimizer.zero_grad()
            # モデルを使用して追加されたノイズを予測
            noise_pred = model(x_t, t)
            # 真のノイズと予測されたノイズとの間の損失を計算（平均二乗誤差）
            loss = F.mse_loss(noise_pred, noise)
            # バックプロパゲーション
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')
        losses.append(avg_loss)

        # 10エポックごとにモデルを保存
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'{model_dir}/model_epoch_{epoch+1}.pth')
            print(f'エポック {epoch+1} でモデルを保存しました')

    # 学習損失のプロット
    plt.plot(range(1, epochs + 1), losses)
    plt.xlabel('エポック')
    plt.ylabel('損失')
    plt.title('学習損失')
    plt.show()

    # 訓練されたモデルを使用して画像を生成
    generated_images = diffuser.sample(model, x_shape=(20, 1, 28, 28))
    # 生成された画像を表示
    show_images(torch.stack([transforms.ToTensor()(img) for img in generated_images]), rows=2, cols=10)
