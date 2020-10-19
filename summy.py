from sumrized import Sumrized
import gensim
import gensim.models.keyedvectors as word2vec
from helper import Helper
import nltk, os

print("asdasd")
lang = "ar"

tools = 'tools/bow/' # in my case, the folder './tools/' contain simlinks
# tools = 'tools/word2vec/'  # in my case, the folder './tools/' contain simlinks

# word2vecEnPath = tools + 'wiki.en/wiki.en.vec'
# word2vecEn = word2vec.KeyedVectors.load_word2vec_format(word2vecEnPath,
#                                                     binary=True,
#                                                    unicode_errors='ignore',
#                                                   limit=50000
#                                                   )


word2vecArPath = tools + 'full_grams_sg_100_wiki.mdl'
# word2vecArPath = tools + 'full_grams_cbow_100_wiki.mdl'

word2vecAr = gensim.models.Word2Vec.load(word2vecArPath)

help = Helper(lang=lang)
testingArticles = [
    'article1.txt',
    'article2.txt',
    'article3.txt',
    'article4.txt',
    'article5.txt',
    'article6.txt'

]
articlePath = 'articles/' + lang + '/' + testingArticles[5]
content = help.getArticleContent(articlePath)

sentences = help.getArticleSentences(content)
summarySize = 10  # [10, 100]
limit = (summarySize * len(sentences)) / 100

sumrized = Sumrized(lang, word2vecAr)
summary = sumrized.summarize(content, limit)

print(summary)


