from utils import pre_process, tt_split, build_rfc_pipeline, fit_predict, pred_cand
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score

df, df_candidates, df_misc = pre_process()

X, y, X_train, X_test, y_train, y_test = tt_split(df)
pipeline = build_rfc_pipeline()
y_pred = fit_predict(pipeline, X_train, y_train, X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification report", classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, pipeline.predict_proba(X_test)[:,1]))

cand_pred, cand_prob = pred_cand(pipeline, df_candidates)
print("Predictions:", cand_pred)
print("Probabilities", cand_prob)

cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
print(cv_scores)
print(f"Mean ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")