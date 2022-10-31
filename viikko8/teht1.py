import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import graphviz
from sklearn.tree import export_graphviz

df = pd.read_csv('iris.csv')

X = df.iloc[:,0:4]
X_org = X
y = df.iloc[:,[4]]

sns.scatterplot(x='petal length (cm)',y='petal width (cm)',hue='Species',data=df)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, 
                                                    random_state = 0)

model = tree.DecisionTreeClassifier(max_depth=4,criterion='gini')
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)

mfi = model.feature_importances_

cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test,y_pred)
print(f'Acc: {score}')
ax = plt.axes()
sns.heatmap(cm, annot=True, fmt='g', ax=ax)
ax.set_title('DT')
plt.show()

dot_data = export_graphviz(
            model,
            out_file=None,
            feature_names = X.columns,
            class_names = df['Class'].unique(),
            filled=True,
            rounded=True)

graph = graphviz.Source(dot_data)
graph.render(filename = 'iris', format = 'png')

Xnew = pd.read_csv('new-iris.csv')

y_pred_new = model.predict(Xnew)
X_pred_new = model.predict_proba(Xnew)
