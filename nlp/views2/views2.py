from django.shortcuts import render
from django.http import HttpResponse,QueryDict
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import MeCab

category_data=pd.read_csv("idx2category.csv")
idx2category={row.k:row.v for idx , row in category_data.iterrows()}
with open("rdmf.pickle",mode="rb") as f:
    model=pickle.load(f)
sample_df=pd.read_csv("cut_products.csv",header=0)
sample_df=pd.DataFrame(sample_df)

tagger = MeCab.Tagger('-Owakati')
corpus = [tagger.parse(sentence).strip() for sentence in sample_df["品名仕様"]]
vectorizer=TfidfVectorizer()
tf_siyou_vec=vectorizer.fit_transform(corpus)


def tf_idf(request):
    if request.method=="GET":
        return render(
            request,
            "nlp/response.html"
        )
    
    else:
        priority_word=[request.POST.getList("level")]
        
        
        return render(
            request,
            "nlp/response.html",
            {"priority_word":priority_word }
        
        )



#Create your views here.
