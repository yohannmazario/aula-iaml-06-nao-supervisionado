# Missão RangoJá — quem são nossos clientes?

Relatório para a Marina (marketing) e para o time de risco.
Aula IAML 06 — Aprendizado de Máquina Não Supervisionado.

## Resumo executivo

Olhamos 398 clientes dos últimos 3 meses, sem nenhum rótulo pronto.
Aparecem **quatro públicos de verdade** e um **quinto grupo de 8 contas** que não se comporta como cliente: pede o tempo todo, no ticket mínimo, e usa cupom em 100% dos pedidos.

A recomendação prática: parar de tratar todo mundo com o mesmo cupom.
Os fiéis não precisam de desconto grande. Os gourmets sumiram e não voltam por cupom. Quem vive de cupom precisa de um teste sem desconto. As 8 contas vão para revisão humana do risco — não para bloqueio automático.

## Personas

Médias do grupo (K-Means, k = 5, dados padronizados).

| Grupo | Qtd | Persona | Principal característica | Campanha para a Marina |
| --- | ---: | --- | --- | --- |
| 0 | 90 | Caçadores de cupom | Ticket baixo (~R$ 32) e cupom em ~84% dos pedidos | Cortar cupom automático. Testar se ainda pedem com desconto menor e ticket mínimo |
| 1 | 120 | Gourmets ocasionais | Ticket alto (~R$ 96), poucos pedidos (~4/mês) e ~10 dias sem pedir | Reativação pela experiência, não pelo desconto: pratos premium e frete em pedido alto |
| 2 | 100 | Fiéis do dia a dia | ~18 pedidos/mês, ticket ~R$ 39, pediram há ~2 dias, pouco cupom (~20%) | Fidelidade (pontos, frete grátis depois de X pedidos). Cupom grande aqui é dinheiro jogado fora |
| 3 | 80 | Corujas da madrugada | ~64% dos pedidos entre 0h e 5h | Cardápio noturno e combos de madrugada. Cupom não é o que segura esse grupo |
| 4 | 8 | Suspeitos de abuso | ~57 pedidos/mês, ticket ~R$ 18, cupom em 100%, pediram hoje | Não é público de marketing. Não mandar mais cupom. Encaminhar ao risco |

Dois componentes do PCA resumem cerca de **70%** da informação (46,6% + 23,3%) e separam esses grupos no gráfico `figuras/03_pca_grupos.png`.

## Contas suspeitas

O Isolation Forest (esperando cerca de 2% de casos fora do padrão) marcou **8 clientes**. Todos coincidem com o grupo 4:

| Cliente | Pedidos/mês | Ticket (R$) | Madrugada (%) | Dias desde o último | Cupom (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| C0184 | 66 | 17,19 | 26,7 | 0 | 100 |
| C0052 | 64 | 19,10 | 33,4 | 0 | 100 |
| C0063 | 62 | 19,08 | 54,4 | 0 | 100 |
| C0373 | 56 | 16,35 | 43,2 | 0 | 100 |
| C0134 | 55 | 16,99 | 41,8 | 1 | 100 |
| C0068 | 53 | 16,34 | 34,6 | 0 | 100 |
| C0049 | 52 | 17,65 | 38,1 | 0 | 100 |
| C0397 | 50 | 17,45 | 31,4 | 0 | 100 |

**Não bloquear essas contas automaticamente.** O padrão é forte e repetido (volume absurdo, ticket no piso, cupom sempre), então vale uma trava temporária de cupom e uma fila de revisão no mesmo dia. Bloqueio total de conta, sem uma pessoa olhar, pune um cliente legítimo no meio de um pedido — e um alarme “quase normal” já apareceu em exemplos de monitoramento de servidor desta aula. Quem decide o que é abuso de verdade é o time de risco, com o histórico da conta na mão.

Se tivéssemos parado em 4 grupos, esses 8 sumiriam dentro dos fiéis. A Marina veria “cliente bom” e mandaria ainda mais benefício.

## Como reproduzir

- Python 3.14.6
- Dependências em `requirements.txt` (`numpy`, `pandas`, `scikit-learn`, `matplotlib`, `scipy`)
- Semente dos dados: `2026` (função `gerar_rangoja`)
- K-Means: `k = 5`, `n_init = 10`, `random_state = 42`, sobre as 5 colunas padronizadas com `StandardScaler`
- Isolation Forest: `contamination = 0.02`, `random_state = 42`
- Registro dos experimentos de k = 2 a 8: [`experimentos.csv`](experimentos.csv)

```bash
pip install -r requirements.txt
python solucao_rangojá.py
```

## Como isso vira produto numa empresa

No primeiro dia ninguém tem o rótulo “fraude”. A detecção sem gabarito aponta as contas estranhas; analistas marcam “abuso” ou “alarme falso”; com esses exemplos, meses depois, um modelo supervisionado pega os golpes já conhecidos — e o modelo sem rótulo continua caçando o golpe novo. O `experimentos.csv`, a semente fixa e o `random_state` são o mínimo para outra pessoa repetir o mesmo resultado. Em produção isso vai para uma ferramenta de registro de experimentos, com revisão humana antes de qualquer bloqueio.

## Respostas curtas das tarefas

**1.** Maior desvio: `pct_cupom` (30,00). Menor: `dias_ultimo` (4,47). Sem padronizar, cupom e ticket dominam a distância; dias desde o último pedido quase não pesa. O histograma de `pedidos_mes` mostra a cauda estranha (50–66 pedidos).

**2.** Depois do `StandardScaler`, cada coluna fica com média 0 e desvio 1.

**3.** Cotovelo e silhueta concordam em **k = 5** (silhueta 0,614, a maior da busca).

**4.** Personas na tabela acima.

**5.** 8 anômalos, todos no grupo 4. Com k = 4 eles entram no grupo dos fiéis e o abuso vira “cliente bom”.

## Uso de assistente de código

O script, os gráficos, o `experimentos.csv` e este texto foram produzidos com um assistente de código no Cursor. A revisão feita em cima da saída:

- os números de inércia, silhueta, tamanho dos grupos e a lista das 8 contas foram conferidos rodando o script de ponta a ponta;
- o k ficou em 5 porque cotovelo e silhueta apontam para lá e porque k = 4 esconde o abuso dentro dos fiéis;
- a recomendação de não bloquear automático segue o combinado da aula: alarme não é sentença.
