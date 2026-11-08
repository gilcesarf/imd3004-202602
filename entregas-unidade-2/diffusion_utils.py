"""Funções auxiliares do Laboratório 9: modelos de difusão em dados 2D.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.preprocessing import StandardScaler
import torch
import gdown
from IPython.display import HTML, SVG, display

BLUE = "#006599"                                                                # Dados reais e processo direto q
ORANGE = "#CC6600"                                                              # Amostras geradas e processo reverso
RED = "#B02020"                                                                 # β_t no cronograma de ruído
GRAY = "#808080"                                                                # Ruído e elementos secundários
LIMIT = 3.5                                                                     # Janela dos quadros: [-3.5, 3.5]²
TIME_CMAP = LinearSegmentedColormap.from_list(                                  # Cinza (t = T) a laranja (t = 0)
    "time", ["#C8C8C8", ORANGE])

# Links dos arquivos no Google Drive
FIGURE_DIR = "figuras"                                                          # Pasta local das figuras
FIGURE_URLS = {                                                                 # Figuras (SVG)
    "processos_difusao":
        "https://drive.google.com/file/d/1Y4xmDVuW1c8QBmowMMlnZzba3ONF6a8v/view?usp=sharing",
    "kernel_difusao":
        "https://drive.google.com/file/d/1LGccwKmuoMf2qVC3DMBW726pyfHT89Kq/view?usp=sharing",
    "rede_ruido":
        "https://drive.google.com/file/d/1cFV62YaTBfXsyEiPLnjQcP2M8hWLctkR/view?usp=sharing",
    "passo_reverso":
        "https://drive.google.com/file/d/1lXnl-LjyzGQSGVovqE5AU47cr7FePeai/view?usp=sharing",
    "treinamento_difusao":
        "https://drive.google.com/file/d/1Ts7G1A1GdqlIvO9bwi3IMGfMJdRALeTf/view?usp=sharing",
    "amostragem_difusao":
        "https://drive.google.com/file/d/1w14_XwpyQ689Hp4a2V-mdUtxPTnlKFoq/view?usp=sharing",
}
ANIMATION_URLS = {                                                              # Animações (HTML)
    "kernel_difusao_animado.html":
        "https://drive.google.com/file/d/1w2YfsxLhuoHl2P7IKfO8bTX-ZFX2FQP5/view?usp=sharing",
}
DATASAURUS_FILE = "DatasaurusDozen-Long.tsv"                                    # Arquivo local do Datasaurus
DATASAURUS_URL = (                                                              # Datasaurus (TSV)
    "https://drive.google.com/file/d/1wG7IYlepeA48UyORo1nP-_cT8oQA4la1/view?usp=sharing"
)


def _download(url, path):
    """Baixa o arquivo do Drive para `path`, se ainda não existir; uma falha só gera um aviso."""
    if os.path.exists(path) or not url:                                         # Já baixado ou sem link
        return
    try:
        # gdown.download(url, path, quiet=True, fuzzy=True) # fuzzy não é reconhecido no meu notebook !!
        gdown.download(url, path, quiet=True)
    except Exception as error:                                                  # Link inválido ou sem rede
        print(f"Falha no download de {path}: {error}")


def show_figure(name):
    """Exibe a figura `name` (SVG) da pasta FIGURE_DIR; baixa o arquivo do Drive, se preciso."""
    os.makedirs(FIGURE_DIR, exist_ok=True)                                      # Cria a pasta, se preciso
    path = os.path.join(FIGURE_DIR, f"{name}.svg")                              # Arquivo da figura
    _download(FIGURE_URLS.get(name), path)                                      # Baixa só se ainda não existir
    try:
        display(SVG(filename=path))                                             # A figura aparece como saída da célula
    except Exception:                                                           # Arquivo ausente ou inválido
        print(f"Figura indisponível: {name}. Confira o link em FIGURE_URLS, no diffusion_utils.py.")


def _time_grid(T, num_frames):
    """Passos inteiros igualmente espaçados entre 0 e T, sem repetição."""
    return np.unique(np.linspace(0, T, num_frames).round().astype(int))


def _frame(ax, points, color, title=None, size=3):
    """Desenha um quadro de pontos 2D, com a janela fixa e sem marcas nos eixos."""
    points = np.asarray(points)                                                 # (N, 2)
    ax.scatter(points[:, 0], points[:, 1], s=size, color=color,                 # Pontos do quadro
               alpha=0.6, linewidths=0)
    ax.set_xlim(-LIMIT, LIMIT)                                                  # Mesma janela em todos os quadros
    ax.set_ylim(-LIMIT, LIMIT)                                                  # Idem no eixo y
    ax.set_aspect("equal")                                                      # Mesma escala nos dois eixos
    ax.set_xticks([])                                                           # Sem marcas no eixo x
    ax.set_yticks([])                                                           # Sem marcas no eixo y
    for spine in ax.spines.values():
        spine.set_color("#D0D0D0")                                              # Moldura discreta
    if title:
        ax.set_title(title, fontsize=12)                                        # Passo t do quadro


def plot_data(data, num_points=1000, seed=42):
    """Plota os dados sobre amostras da distribuição a priori N(0, I)."""
    torch.manual_seed(seed)                                                     # Fixa as amostras da priori
    prior = torch.randn(num_points, 2).numpy()                                  # x_T ~ N(0, I)
    points = data[:num_points].cpu().numpy()                                    # Pontos x_0 do conjunto
    plt.figure(figsize=(5, 5))                                                  # Tamanho da figura
    plt.scatter(prior[:, 0], prior[:, 1], s=6, alpha=0.5, color=GRAY,           # Amostras da priori
                label=r"$\mathcal{N}(\mathbf{0}, \mathbf{I})$")
    plt.scatter(points[:, 0], points[:, 1], s=6, alpha=0.8, color=BLUE,         # Pontos do conjunto
                label=r"dados $\mathbf{x}_0$")
    plt.xlim(-LIMIT, LIMIT)                                                     # Mesma janela dos quadros
    plt.ylim(-LIMIT, LIMIT)                                                     # Idem no eixo y
    plt.gca().set_aspect("equal")                                               # Mesma escala nos dois eixos
    plt.legend(loc="upper right")                                               # Legenda
    plt.title("Dados e distribuição a priori")                                  # Título
    plt.tight_layout()                                                          # Ajusta as margens
    plt.show()                                                                  # Exibe a figura


# def show_animation(name):
#     """Exibe a animação salva no arquivo HTML `name`; baixa o arquivo do Drive, se preciso."""
#     _download(ANIMATION_URLS.get(name), name)                                   # Baixa só se ainda não existir
#     if os.path.exists(name):
#         display(HTML(filename=name))                                            # A animação aparece como saída
#     else:                                                                       # Arquivo ausente
#         print(f"Animação indisponível: {name}. Confira o link em ANIMATION_URLS, no diffusion_utils.py.")

def show_animation(name):
    """Exibe a animação HTML isolada em um iframe."""
    from html import escape

    _download(ANIMATION_URLS.get(name), name)  # Baixa se necessário

    if not os.path.exists(name):
        print(
            f"Animação indisponível: {name}. "
            "Confira o link em ANIMATION_URLS, no diffusion_utils.py."
        )
        return

    with open(name, "r", encoding="utf-8") as arquivo:
        conteudo = arquivo.read()

    iframe = (
        '<iframe '
        'sandbox="allow-scripts" '
        'style="width:100%; height:650px; border:0; background:white;" '
        f'srcdoc="{escape(conteudo, quote=True)}">'
        '</iframe>'
    )

    display(HTML(iframe))

def plot_noise_schedule(noise_scheduler):
    """Plota os coeficientes por passo; α_t do sinal e (1-α_t) do ruído em três gráficos de linhas."""
    T = noise_scheduler.T                                                       # Número de passos
    t = np.arange(T + 1)                                                        # Passos 0, 1, ..., T
    betas = noise_scheduler.betas.cpu().numpy()                                 # β_t
    alphas = noise_scheduler.alphas.cpu().numpy()                               # α_t
    alphas_bar = noise_scheduler.alphas_bar.cpu().numpy()                       # ᾱ_t

    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.4))                          # Três painéis lado a lado

    axs[0].plot(t, betas, color=RED,                                            # β_t: ruído por passo
                label=r"$\beta_t$")
    axs[0].plot(t, alphas, "--", color=BLUE,                                    # α_t: sinal mantido por passo
                label=r"$\alpha_t = (1 - \beta_t)$")
    axs[0].set_ylim(0, 1.10)
    axs[0].legend(loc="center right")
    axs[0].set_title("Ruído adicionado em cada passo")                          # Título do painel

    axs[1].plot(t, alphas_bar, color="black",                                   # Sinal acumulado
                label=r"$\bar{\alpha}_t$")
    axs[1].set_ylim(0, 1)
    axs[1].legend(loc="upper right")
    axs[1].set_title("Sinal")                                                   # Título do painel

    axs[2].plot(t, 1 - alphas_bar, color="black",                               # Ruído acumulado
                label=r"$(1 - \bar{\alpha}_t)$")
    axs[2].set_ylim(0, 1)
    axs[2].legend(loc="lower right")
    axs[2].set_title("Ruído")                                                   # Título do painel

    for ax in axs:
        ax.set_xlim(0, T)                                                       # Intervalo completo de passos
        ax.set_xlabel(r"$t$")                                                   # Passo de tempo
        ax.grid(True, alpha=0.45)                                               # Grade discreta

    plt.tight_layout()                                                          # Evita sobreposições
    plt.show()                                                                  # Exibe a figura


def plot_time_encoding(time_enc, title=None):
    """Plota o mapa de calor da codificação temporal."""
    table = time_enc.time_encodings.cpu().numpy()                               # E, shape (T+1, D)
    T, D = table.shape[0] - 1, table.shape[1]                                   # Passos e dimensão

    fig, ax = plt.subplots(figsize=(8, 4.2))                                    # Figura única
    image = ax.imshow(table.T, aspect="auto", origin="lower",                   # Componente x passo
                      vmin=-1, vmax=1, interpolation="nearest",
                      extent=[-0.5, T + 0.5, -0.5, D - 0.5])

    fig.colorbar(image, ax=ax, label="valor da componente")                     # Barra de cores
    ax.set_xlabel(r"passo $t$")                                                 # Rótulo do eixo x
    ax.set_ylabel("componente")                                                 # Rótulo do eixo y
    default_title = rf"Codificação temporal $\mathbf{{e}}(t)$, $D = {D}$"       # Título padrão
    ax.set_title(title or default_title)                                        # Título

    plt.tight_layout()                                                          # Ajusta as margens
    plt.show()                                                                  # Exibe a figura


def plot_activation():
    """Compara a SiLU com a ReLU."""
    x = torch.linspace(-6, 6, 200)                                              # Valores de x em [-6, 6]
    plt.figure(figsize=(4.4, 3.2))                                              # Tamanho da figura
    plt.plot(x.numpy(), torch.nn.functional.silu(x).numpy(), color=ORANGE,      # Curva da SiLU
             label=r"$\mathrm{SiLU}(x) = x\,\sigma(x)$")
    plt.plot(x.numpy(), torch.relu(x).numpy(), "--", color=GRAY,                # ReLU, para comparação
             label=r"$\mathrm{ReLU}(x)$")
    plt.xlabel(r"$x$")                                                          # Rótulo do eixo x
    plt.grid(True, alpha=0.3)                                                   # Grade discreta
    plt.legend()                                                                # Legenda
    plt.tight_layout()                                                          # Ajusta as margens
    plt.show()                                                                  # Exibe a figura


def compare_processes(model, data, num_frames=6, num_points=1000, seed=42, title=None):
    """Quadros de q(x_t) (acima) e de p_theta(x_t) (abaixo) nos mesmos passos t."""
    noise_scheduler = model.noise_scheduler                                     # Cronograma do modelo
    T = noise_scheduler.T                                                       # Número de passos
    times = _time_grid(T, num_frames)                                           # Passos exibidos: 0, ..., T
    device = noise_scheduler.betas.device                                       # Dispositivo do modelo
    x0 = data[:num_points].to(device)                                           # Pontos limpos x_0
    torch.manual_seed(seed)                                                     # Fixa ε, x_T e os z
    noise = torch.randn_like(x0)                                                # Ruído do processo direto
    _, snaps = model.reverse_diffusion_step_loop(                               # x_t do processo reverso
        num_points, times_to_show=set(times.tolist()))
    n = len(times)                                                              # Número de colunas
    fig, axs = plt.subplots(2, n, figsize=(2.05 * n, 4.7))                      # Duas linhas de quadros
    for k, t in enumerate(times):
        t_batch = torch.full((len(x0),), int(t), dtype=torch.long,              # Mesmo t para todos os pontos
                             device=device)
        x_t = noise_scheduler.forward_diffusion(x0, t_batch, noise).cpu()       # x_t ~ q(x_t | x_0)
        _frame(axs[0, k], x_t, BLUE, rf"$t = {t}$")                             # Processo direto
        _frame(axs[1, k], snaps[int(t)], ORANGE)                                # Processo reverso
    axs[0, 0].set_ylabel(r"direto: $q(\mathbf{x}_t)$", fontsize=12)             # Rótulo da linha de cima
    axs[1, 0].set_ylabel(r"reverso: $p_\theta(\mathbf{x}_t)$", fontsize=12)     # Rótulo da linha de baixo
    fig.text(0.99, 0.5, r"$t$ crescente: adiciona ruído $\rightarrow$",         # Sentido de q
             ha="right", va="center", color=BLUE, fontsize=11)
    fig.text(0.01, 0.02, r"$\leftarrow$ $t$ decrescente: remove ruído",         # Sentido de p_theta
             ha="left", va="bottom", color=ORANGE, fontsize=11)
    if title:
        fig.suptitle(title)                                                     # Identifica a execução
    plt.tight_layout(rect=(0, 0.04, 1, 1), h_pad=2.2)                           # Espaço para as setas
    plt.show()                                                                  # Exibe a figura


def plot_trajectories(model, data, num_trajectories=40, seed=42, title=None):
    """Caminhos x_T, ..., x_0 de alguns pontos no processo reverso, coloridos pelo passo t."""
    T = model.noise_scheduler.T                                                 # Número de passos
    torch.manual_seed(seed)                                                     # Fixa x_T e os z
    steps = list(range(T, -1, -max(1, T // 100)))                               # T, ..., 0 a cada T/100 passos
    steps += [] if steps[-1] == 0 else [0]                                      # Termina em t = 0
    _, snaps = model.reverse_diffusion_step_loop(                               # x_t de cada ponto
        num_trajectories, times_to_show=set(steps))
    paths = np.stack([snaps[t].numpy() for t in steps])                         # (passos, n, 2), de T a 0
    colors = TIME_CMAP(np.linspace(0, 1, len(steps) - 1))                       # Uma cor por segmento
    colors[:, 3] = np.linspace(0.12, 0.95, len(colors))                         # Opacidade cresce até t = 0
    fig, ax = plt.subplots(figsize=(6.2, 5.2))                                  # Tamanho da figura
    background = data[:3000].cpu().numpy()                                      # Dados, ao fundo
    ax.scatter(background[:, 0], background[:, 1], s=3, color=BLUE,             # Curva dos dados
               alpha=0.35, zorder=2.5, linewidths=0, label="dados")
    for k in range(num_trajectories):
        segments = np.stack([paths[:-1, k], paths[1:, k]], axis=1)              # (passos - 1, 2, 2)
        ax.add_collection(LineCollection(segments, colors=colors,               # Caminho do ponto k
                                         linewidths=1.1))
    ax.scatter(paths[0, :, 0], paths[0, :, 1], s=16, color=GRAY,                # Pontos de partida
               zorder=3, label=r"início: $\mathbf{x}_T$")
    ax.scatter(paths[-1, :, 0], paths[-1, :, 1], s=16, color=ORANGE,            # Pontos de chegada
               zorder=3, label=r"fim: $\mathbf{x}_0$")
    mappable = ScalarMappable(norm=Normalize(0, T), cmap=TIME_CMAP.reversed())  # Cores de t = 0 a T
    cbar = fig.colorbar(mappable, ax=ax, ticks=[0, T])                          # Barra de cores
    cbar.set_label(r"passo $t$")                                                # Rótulo da barra
    ax.set_xlim(-LIMIT, LIMIT)                                                  # Mesma janela dos quadros
    ax.set_ylim(-LIMIT, LIMIT)                                                  # Idem no eixo y
    ax.set_aspect("equal")                                                      # Mesma escala nos dois eixos
    ax.legend(loc="upper right", fontsize=9)                                    # Legenda
    ax.set_title(title or "Trajetórias do processo reverso")                    # Título
    plt.tight_layout()                                                          # Ajusta as margens
    plt.show()                                                                  # Exibe a figura


def plot_training_curve(history, title=None):
    """Plota a perda por época de um treinamento."""
    plt.figure(figsize=(6, 3.6))                                                # Tamanho da figura
    plt.plot(history.loss, color=ORANGE,                                        # Perda média de cada época
             label=r"perda: $\| \epsilon - \epsilon_\theta(\mathbf{x}_t, t) \|^2$")
    plt.xlabel("Época")                                                         # Rótulo do eixo x
    plt.grid(True, alpha=0.3)                                                   # Grade discreta
    plt.legend()                                                                # Legenda
    plt.title(title or "Curva de treinamento")                                  # Título
    plt.tight_layout()                                                          # Ajusta as margens
    plt.show()                                                                  # Exibe a figura


def compare_training_curves(histories, symbol, title=None):
    """Sobrepõe as curvas de perda de várias execuções, com legenda {valor: histórico}."""
    plt.figure(figsize=(6.4, 3.8))                                              # Tamanho da figura
    for value, history in histories.items():
        plt.plot(history.loss, label=rf"${symbol} = {value}$")                  # Uma curva por execução
    plt.xlabel("Época")                                                         # Rótulo do eixo x
    plt.ylabel("perda")                                                         # Rótulo do eixo y
    plt.grid(True, alpha=0.3)                                                   # Grade discreta
    plt.legend()                                                                # Legenda
    plt.title(title or rf"Curvas de perda para cada valor de ${symbol}$")       # Título
    plt.tight_layout()                                                          # Ajusta as margens
    plt.show()                                                                  # Exibe a figura


def animate_reverse_process(model, num_points=1000, num_frames=50, seed=42):
    """Animação do processo reverso, de x_T ~ N(0, I) até x_0."""
    T = model.noise_scheduler.T                                                 # Número de passos
    times = _time_grid(T, num_frames)[::-1]                                     # Passos exibidos: T, ..., 0
    torch.manual_seed(seed)                                                     # Fixa x_T e os z
    _, snaps = model.reverse_diffusion_step_loop(                               # x_t dos passos exibidos
        num_points, times_to_show=set(times.tolist()))
    fig, ax = plt.subplots(figsize=(4.4, 4.4))                                  # Tamanho da figura
    _frame(ax, snaps[int(times[0])], ORANGE, size=5)                            # Primeiro quadro: x_T
    points = ax.collections[0]                                                  # Pontos a atualizar

    def update(k):
        points.set_offsets(snaps[int(times[k])].numpy())                        # Pontos do passo t
        ax.set_title(rf"$p_\theta(\mathbf{{x}}_{{{times[k]}}})$", fontsize=12)  # Passo t
        return (points,)

    movie = animation.FuncAnimation(fig, update, frames=len(times),             # Quadros da animação
                                    interval=120)
    plt.close(fig)                                                              # Evita exibir a figura estática
    return HTML(movie.to_jshtml())                                              # Animação com controles


def _datasaurus_table():
    """Tabela do Datasaurus (colunas dataset, x e y); baixa o arquivo do Drive, se preciso."""
    _download(DATASAURUS_URL, DATASAURUS_FILE)                                  # Baixa só se ainda não existir
    return pd.read_csv(DATASAURUS_FILE, sep="\t")                               # Carrega o arquivo TSV


def load_datasaurus(num_samples, dataset, seed=42):
    """Carrega um conjunto do Datasaurus, com as coordenadas padronizadas."""
    df = _datasaurus_table()                                                    # Carrega o arquivo TSV
    names = df["dataset"].unique()                                              # Conjuntos disponíveis
    if dataset not in names:
        raise ValueError(f"Conjunto desconhecido: {dataset}. Opções: {list(names)}")

    X = df.loc[df["dataset"] == dataset, ["x", "y"]].to_numpy()                 # Seleciona o conjunto
    X = StandardScaler().fit_transform(X)                                       # Média 0, variância 1

    rng = np.random.default_rng(seed)                                           # Gerador reprodutível
    indices = rng.choice(len(X), size=num_samples,                              # Seleciona os pontos
                         replace=num_samples > len(X))
    X = X[indices]

    return torch.tensor(X, dtype=torch.float32)                                 # Tensor (num_samples, 2)


def plot_datasaurus(num_cols=5):
    """Grade de gráficos de dispersão dos conjuntos do Datasaurus, padronizados como em load_datasaurus."""
    table = _datasaurus_table()                                                 # Colunas dataset, x e y
    names = table["dataset"].unique()                                           # Conjuntos, na ordem do arquivo
    num_rows = -(-len(names) // num_cols)                                       # Linhas da grade (arredonda para cima)
    fig, axs = plt.subplots(num_rows, num_cols,                                 # Um quadro por conjunto
                            figsize=(2.3 * num_cols, 2.3 * num_rows))
    for ax, name in zip(axs.flat, names):
        points = table.loc[table["dataset"] == name, ["x", "y"]].to_numpy()     # Pontos do conjunto
        points = StandardScaler().fit_transform(points)                         # Média 0, variância 1
        _frame(ax, points, BLUE, f'"{name}"', size=6)                           # Quadro com o nome do conjunto
    for ax in axs.flat[len(names):]:
        ax.axis("off")                                                          # Esconde os quadros vazios
    plt.tight_layout()                                                          # Ajusta as margens
    plt.show()                                                                  # Exibe a figura
