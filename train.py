# coding:utf-8
import os
import urllib
import html
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class url(object):
    def __init__(self):
        #读取数据
        good_query_list = self.get_query_list('goodqueries.txt')
        bad_query_list = self.get_query_list('badqueries.txt')

        #给黑、白数据分别打标签
        good_y = [0 for i in range(0, len(good_query_list))]
        bad_y = [1 for i in range(0, len(bad_query_list))]

        queries = good_query_list + bad_query_list
        y = good_y + bad_y

        #将原始文本数据分割转化成向量
        self.vectorizer = TfidfVectorizer(tokenizer=self.get_ngrams)

        #把文本字符串转化成（[i,j],Tfidf值）矩阵X
        X = self.vectorizer.fit_transform(queries)

        #分割训练数据（建立模型）和测试数据（测试模型准确度）
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=42)

        #定义模型训练方法（逻辑回归）
        self.lgs = PMMLPipeline([('LogisticModer', LogisticRegression(solver='liblinear'))])

        #训练模型
        self.lgs.fit(X_train, y_train)

        #测试模型准确度
        print('模型准确度:{}'.format(self.lgs.score(X_test, y_test)))

        sklearn2pmml(self.lgs, '.\lgs.pmml', with_repr=True)

    #获取文本中的请求列表
    def get_query_list(self, filename):
        directory = str(os.getcwd()) + '\\data\\test'
        filepath = directory + '\\' + filename
        data = open(filepath, 'r', encoding = 'utf-8').readlines()
        query_list = []
        for d in data:
            d = str(urllib.parse.unquote(d))
            query_list.append(d)
        return list(set(query_list))

    #分割字符串，每3个字符作一次分割
    def get_ngrams(self, query):
        tempQuery = str(query)
        ngrams = []
        for i in range(0, len(tempQuery)-3):
            ngrams.append(tempQuery[i:i+3])
        return ngrams

    #预测新的url
    def predict(self, newQueries):
        newQueries = [urllib.parse.unquote(url) for url in newQueries]
        X_predict = self.vectorizer.transform(newQueries)
        res = self.lgs.predict(X_predict)
        res_list = []
        for q,r in zip(newQueries, res):
            tmp = '正常请求' if r == 0 else '恶意请求'
            q_entity = html.escape(q)
            res_list.append({'url':q_entity, 'res':tmp})
        print("预测的结果列表:{}".format(str(res_list)))
        return res_list

if __name__ == '__main__':
    u = url()

    u.predict(['www.foo.com/id=1<script>alert(1)</script>', 'www.foo.com/name=admin\' or 1=1', 'abc.com/admin.php',
               '"><svg onload=confirm(1)>', 'test/q=<a href="javascript:confirm(1)>', 'q=../etc/passwd'])