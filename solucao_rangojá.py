"""
Missão RangoJá — Aula IAML 06 (aprendizado não supervisionado).

Descobre tipos de cliente sem rótulos, escolhe o número de grupos e
aponta contas que podem estar abusando de cupom.

Como rodar:
    pip install -r requirements.txt
    python solucao_rangojá.py

Gera:
    experimentos.csv
    figuras/01_histogramas.png
    figuras/02_cotovelo_silhueta.png
    figuras/03_pca_grupos.png
"""

import os
import sys

import matplotlib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Código base da aula — gerador dos dados (não alterar; semente = 2026)
# ---------------------------------------------------------------------------


def gerar_rangoja(seed=2026):
    rng = np.random.default_rng(seed)

    def persona(n, ped, tic, madr, dias, cup):
        return pd.DataFrame(
            {
                "pedidos_mes": np.clip(rng.normal(*ped, n), 1, None).round(0),
                "ticket_medio": np.clip(rng.normal(*tic, n), 15, None).round(2),
                "pct_madrugada": np.clip(rng.normal(*madr, n), 0, 100).round(1),
                "dias_ultimo": np.clip(rng.normal(*dias, n), 0, None).round(0),
                "pct_cupom": np.clip(rng.normal(*cup, n), 0, 100).round(1),
            }
        )

    df = pd.concat(
        [
            persona(120, (4, 1.5), (95, 15), (5, 3), (10, 4), (15, 8)),
            persona(100, (18, 4), (38, 6), (8, 4), (2, 1.5), (20, 8)),
            persona(80, (10, 3), (45, 8), (65, 10), (3, 2), (25, 10)),
            persona(90, (6, 2), (32, 5), (10, 5), (6, 3), (85, 8)),
            persona(8, (60, 8), (18, 2), (40, 15), (0, 0.5), (100, 0)),
        ],
        ignore_index=True,
    )
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "cliente_id", [f"C{i:04d}" for i in range(1, len(df) + 1)])
    return df


FEATURES = [
    "pedidos_mes",
    "ticket_medio",
    "pct_madrugada",
    "dias_ultimo",
    "pct_cupom",
]
K_ESCOLHIDO = 5
PASTA_FIGURAS = "figuras"

# Nomes dados depois de olhar o perfil médio de cada grupo (k = 5, random_state = 42).
PERSONAS = {
    0: "Caçadores de cupom",
    1: "Gourmets ocasionais",
    2: "Fiéis do dia a dia",
    3: "Corujas da madrugada",
    4: "Suspeitos de abuso",
}


def secao(titulo):
    print(f"\n{'=' * 72}\n{titulo}\n{'=' * 72}")


def main():
    os.makedirs(PASTA_FIGURAS, exist_ok=True)
    df = gerar_rangoja()

    # ----- Tarefa 1: exploração ------------------------------------------------
    secao("Tarefa 1 — Conheça os dados")
    print(f"Base: {df.shape[0]} clientes, {df.shape[1]} colunas")
    print("\n1.1 Estatísticas descritivas")
    print(df[FEATURES].describe().round(2).to_string())

    desvios = df[FEATURES].std().sort_values(ascending=False)
    print("\n1.2 Desvio padrão (maior → menor)")
    print(desvios.round(2).to_string())

    eixos = df[FEATURES].hist(figsize=(12, 8), bins=20, color="#FF5000", edgecolor="white")
    plt.suptitle("Distribuição de cada variável — RangoJá")
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_FIGURAS, "01_histogramas.png"), dpi=120)
    plt.close()
    print(f"\n1.3 Histogramas salvos em {PASTA_FIGURAS}/01_histogramas.png")

    print(
        """
Respostas
a) Maior desvio padrão: pct_cupom (30,00). Menor: dias_ultimo (4,47).
b) Sem padronizar, pct_cupom e ticket_medio mandam no K-Means: a distância
   euclidiana soma diferenças brutas, e essas colunas variam dezenas de
   unidades. dias_ultimo fica praticamente ignorada — a distância entre
   clientes é de poucos dias, minúscula perto de R$ 50 no ticket ou 40
   pontos percentuais de cupom.
c) O histograma de pedidos_mes tem uma cauda pequena e distante (por volta
   de 50 a 66 pedidos/mês, enquanto a mediana é 7). É o grupo estranho.
"""
    )

    # ----- Tarefa 2: padronização ---------------------------------------------
    secao("Tarefa 2 — Padronize")
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])
    prova = pd.DataFrame(
        {"media": X.mean(axis=0), "desvio": X.std(axis=0)},
        index=FEATURES,
    )
    print(prova.round(2).to_string())
    print("Médias ~ 0 e desvios ~ 1: a padronização funcionou.")

    # ----- Tarefa 3: escolher k -----------------------------------------------
    secao("Tarefa 3 — Quantos tipos de cliente existem?")
    registro = []
    inercias, silhuetas, ks = [], [], list(range(2, 9))
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        silhueta = float(silhouette_score(X, km.labels_))
        inercias.append(km.inertia_)
        silhuetas.append(silhueta)
        registro.append(
            {
                "experimento": f"kmeans_k{k}",
                "algoritmo": "KMeans",
                "k": k,
                "inercia": round(km.inertia_, 1),
                "silhueta": round(silhueta, 3),
            }
        )
        print(f"k={k} | inércia={km.inertia_:8.1f} | silhueta={silhueta:.3f}")

    log = pd.DataFrame(registro)
    log.to_csv("experimentos.csv", index=False)
    print("\nRegistro salvo em experimentos.csv")
    print(log.to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ks, inercias, "o-", color="#FF5000")
    ax[0].set_title("Cotovelo")
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("Inércia")
    ax[1].plot(ks, silhuetas, "o-", color="#1a1a1a")
    ax[1].set_title("Silhueta")
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("Silhueta")
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_FIGURAS, "02_cotovelo_silhueta.png"), dpi=120)
    plt.close()

    print(
        f"""
Respostas
a) Cotovelo: k = {K_ESCOLHIDO}. A inércia cai forte até k = 5
   (1241 → 832 → 515 → 306) e, a partir daí, a queda fica pequena
   (306 → 237 → 213 → 195). O braço "deita" depois de 5.
b) Silhueta: k = {K_ESCOLHIDO}. É o pico (0,614). Em k = 4 já é 0,591
   e em k = 6 cai para 0,557.
c) Os dois concordam: k = {K_ESCOLHIDO}. Essa escolha separa um grupo
   minúsculo que some se usarmos k = 4 — isso aparece na Tarefa 5.
"""
    )

    # ----- Tarefa 4: personas -------------------------------------------------
    secao("Tarefa 4 — Personas")
    modelo = KMeans(n_clusters=K_ESCOLHIDO, n_init=10, random_state=42)
    df["grupo"] = modelo.fit_predict(X)

    perfil = df.groupby("grupo")[FEATURES].mean().round(1)
    perfil.insert(0, "clientes", df.groupby("grupo").size())
    perfil["persona"] = perfil.index.map(PERSONAS)
    print(perfil.to_string())

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    print(
        f"\nVariância explicada: PC1={var[0]:.3f}  PC2={var[1]:.3f}  "
        f"total={var.sum():.3f}"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    for grupo, nome in PERSONAS.items():
        mascara = df["grupo"] == grupo
        ax.scatter(
            coords[mascara, 0],
            coords[mascara, 1],
            s=28,
            label=f"{grupo}: {nome} (n={int(mascara.sum())})",
        )
    ax.set_xlabel(f"PC1 ({var[0]:.0%})")
    ax.set_ylabel(f"PC2 ({var[1]:.0%})")
    ax.set_title("Clientes da RangoJá em 2 dimensões")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_FIGURAS, "03_pca_grupos.png"), dpi=120)
    plt.close()

    # ----- Tarefa 5: abuso de cupom -------------------------------------------
    secao("Tarefa 5 — Abuso de cupom")
    floresta = IsolationForest(contamination=0.02, random_state=42)
    df["anomalia"] = floresta.fit_predict(X)

    suspeitos = df.loc[df["anomalia"] == -1, ["cliente_id", *FEATURES, "grupo"]].sort_values(
        "pedidos_mes", ascending=False
    )
    print(f"Clientes marcados como anômalos: {len(suspeitos)}")
    print(suspeitos.to_string(index=False))
    print("\nCruzamento grupo × anomalia (−1 = suspeito)")
    print(pd.crosstab(df["grupo"], df["anomalia"]))

    # O que aconteceria com k = 4: os 8 suspeitos somem dentro dos fiéis.
    rotulos_k4 = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(X)
    df["grupo_k4"] = rotulos_k4
    print("\nSe k = 4, onde caem os anômalos?")
    print(pd.crosstab(df.loc[df["anomalia"] == -1, "grupo_k4"], df.loc[df["anomalia"] == -1, "anomalia"]))

    print(
        """
Respostas
a) 8 clientes. Em comum: 50 a 66 pedidos/mês, ticket no piso (cerca de
   R$ 16 a R$ 19), cupom em 100% dos pedidos e último pedido hoje ou ontem.
b) Com k = 5, os 8 caem todos no grupo 4 e em nenhum outro. Isso confirma
   a Tarefa 3c: o quinto grupo não é um capricho do gráfico, é o padrão
   de abuso que a Marina pediu para o time de risco.
c) Com k = 4 eles entram no grupo dos fiéis do dia a dia (o grupo que tem
   muitos pedidos e ticket baixo). O risco é a Marina tratar abuso como
   fidelidade: mais cupom e frete grátis justamente para quem só pede
   porque o desconto cobre o pedido inteiro.
"""
    )

    print("Arquivos gerados: experimentos.csv e figuras/*.png")
    return df, perfil


if __name__ == "__main__":
    main()
