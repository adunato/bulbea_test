import os

os.environ["BULBEA_QUANDL_API_KEY"] = 'VGjjpcct_DscJ4Fa8DCP'
import bulbea as bb

share = bb.Share(source='SSE', ticker='MSFT', provider='alphavantage')
# share = bb.Share(source='SSE', ticker='AMZ', provider='quandl')
share.groupDataByAttribute()
'''print(share.data)'''
from bulbea.learn.evaluation import split

import numpy as np

Xtrain, Xtest, ytrain, ytest = split(share, '4. close', normalize=True)
# Xtrain, Xtest, ytrain, ytest = split(share, 'Last', normalize=True)

Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))
Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))

from bulbea.learn.models import RNN

rnn = RNN([1, 100, 100, 1])  # number of neurons in each layer
rnn.fit(Xtrain, ytrain)

from sklearn.metrics import mean_squared_error

p = rnn.predict(Xtest)
mean_squared_error(ytest, p)
import matplotlib.pyplot as pplt

pplt.plot(ytest)
pplt.plot(p)
pplt.show()
