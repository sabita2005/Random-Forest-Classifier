#parameter Tunning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X=df.iloc[:,[2,3]].values
y=df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
ac
#0.925
bias=classifier.score(X_train,y_train)
bias
#0.996875
variance=classifier.score(X_test,y_test)
variance
#0.9925
#------------------------------------------*****----------------------------------------------------
#Hyper paramete Tunning:1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X=df.iloc[:,[2,3]].values
y=df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=50)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
ac
#0.9375
bias=classifier.score(X_train,y_train)
bias
#0.996875
variance=classifier.score(X_test,y_test)
variance
#0.9375
#--------------------------------------------------------------------------------------------------
#Hyper paramete Tunning:2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X=df.iloc[:,[2,3]].values
y=df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=75)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
ac
#0.9375
bias=classifier.score(X_train,y_train)
bias
#0.996875
variance=classifier.score(X_test,y_test)
variance
#0.9375
#--------------------------------------------------------------------------------------------------
#Hyper paramete Tunning:3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X=df.iloc[:,[2,3]].values
y=df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100,criterion="entropy")
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
ac
#0.9125
bias=classifier.score(X_train,y_train)
bias
#0.996875
variance=classifier.score(X_test,y_test)
variance
#0.9125
#--------------------------------------------------------------------------------------------------
#Hyper paramete Tunning:4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X=df.iloc[:,[2,3]].values
y=df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100,criterion="log_loss")
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
ac
#0.95
bias=classifier.score(X_train,y_train)
bias
#0.996875
variance=classifier.score(X_test,y_test)
variance
#0.95
#--------------------------------------------------------------------------------------------------
#Hyper paramete Tunning:2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv(r"D:\Python Home Work\ML programming\Social_Network_Ads.csv")
X=df.iloc[:,[2,3]].values
y=df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=75)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
ac
#0.925
bias=classifier.score(X_train,y_train)
bias
#0.996875
variance=classifier.score(X_test,y_test)
variance
#0.925