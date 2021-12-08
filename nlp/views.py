from django.shortcuts import render
from django.http import HttpResponse ,QueryDict
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import MeCab
from bs4 import BeautifulSoup


category_data=pd.read_csv("idx2category.csv")
idx2category={row.k:row.v for idx , row in category_data.iterrows()}
with open("rdmf.pickle",mode="rb") as f:
    model=pickle.load(f)
sample_df=pd.read_csv("all_products.csv",header=0)
sample_df=pd.DataFrame(sample_df)

tagger = MeCab.Tagger('-Owakati')
corpus = [tagger.parse(sentence).strip() for sentence in sample_df["品名仕様"]]
vectorizer=TfidfVectorizer()
tf_siyou_vec=vectorizer.fit_transform(corpus)

def index(request):
    if request.method=="GET":
        return render(
            request,
            "nlp/home.html"
        )
    elif request.method=="POST": 
        if 'tf_idf' in request.POST:
            priority_word=request.POST.getlist("level")
            text=" ".join(priority_word)
            text_tf = vectorizer.transform([tagger.parse(text).strip()])
            similarity =cosine_similarity(text_tf, tf_siyou_vec)[0]
            topn_indices = np.argsort(similarity)[::-1][:10]
            title=request.POST.get("input_indno")
            
            topn_df=sample_df.loc[[int(title)]]
            for i in topn_indices:    
                x=sample_df.loc[[i]]
                topn_df=topn_df.append(x)
           
            context=topn_df.to_html()
            return render(
                request,
                "nlp/result.html",
                {"products":context}
            )
        else:

            try:
            

                title=request.POST.getlist("title")
                input_indno=sample_df.index[sample_df["発注コード"]==int(title[0])].tolist()
                product_list=sample_df["仕様"][input_indno[0]]
                product_name=sample_df["品名"][input_indno[0]]
                pro_list=[product_list]
                result= model.predict(pro_list)[0]
                pred=idx2category[result]
                wakati_word=product_list.split()
                #word={"word":(for word in wakati_word)} 
                input_indno=input_indno[0]       

                return render(
                    request,
                    "nlp/response.html",
                    {"product_name":product_name,"title": product_list,"category":pred,
                    "wakati_word":wakati_word,"input_indno":input_indno}
                )
            except ValueError as error:
                error_word="入力された発注コードは登録されていません。"
                return render(
                    request,
                    "nlp/home.html",
                    {"product_name":error_word}
                )

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
