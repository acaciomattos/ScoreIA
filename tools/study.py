from duckduckgo_search import DDGS
import newspaper

def buscar_referencias(tema, max_resultados=5):
    resultados = []
    with DDGS() as ddgs:
        for r in ddgs.text(tema, max_results=max_resultados):
            resultados.append({
                "titulo": r["title"],
                "url": r["href"]
            })
    return resultados


def extrair_texto(url):
    artigo = newspaper.Article(url)
    artigo.download()
    artigo.parse()
    return artigo.text[:3000]
