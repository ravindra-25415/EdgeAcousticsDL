import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,roc_auc_score,confusion_matrix,roc_curve,precision_recall_curve

def optimize_threshold(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    thresholds=np.unique(np.r_[0,np.linspace(0.01,.99,99),p,1])
    # Only validation labels/predictions may be supplied.
    from sklearn.metrics import f1_score
    scores=[f1_score(y,p>=t,zero_division=0) for t in thresholds]
    return float(thresholds[np.argmax(scores)])

def summarize(y_anomaly,p_anomaly,y_machine,p_machine,threshold=.5):
    ya=np.asarray(y_anomaly,dtype=int);pa=np.asarray(p_anomaly)
    ym=np.asarray(y_machine,dtype=int);pm=np.asarray(p_machine)
    pred_a=(pa>=threshold).astype(int);pred_m=pm.argmax(axis=1)
    ap,ar,af,_=precision_recall_fscore_support(ya,pred_a,average='binary',zero_division=0)
    mp,mr,mf,_=precision_recall_fscore_support(ym,pred_m,average='macro',zero_division=0)
    tn,fp,fn,tp=confusion_matrix(ya,pred_a,labels=[0,1]).ravel()
    try:auc=float(roc_auc_score(ya,pa))
    except ValueError:auc=None
    return dict(n=len(ya),threshold=float(threshold),anomaly_accuracy=float(accuracy_score(ya,pred_a)),
     anomaly_precision=float(ap),anomaly_recall=float(ar),anomaly_f1=float(af),anomaly_auc=auc,
     false_alarm_rate=float(fp/max(1,fp+tn)),missed_anomaly_rate=float(fn/max(1,fn+tp)),
     machine_accuracy=float(accuracy_score(ym,pred_m)),machine_macro_precision=float(mp),
     machine_macro_recall=float(mr),machine_macro_f1=float(mf),joint_f1=float((af+mf)/2),
     anomaly_confusion_matrix=[[int(tn),int(fp)],[int(fn),int(tp)]],
     machine_confusion_matrix=confusion_matrix(ym,pred_m,labels=[0,1,2,3]).tolist())

def plot_metrics(y,p,ym,pm,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if len(np.unique(y))>1:
        fpr,tpr,_=roc_curve(y,p);fig,ax=plt.subplots();ax.plot(fpr,tpr);ax.set(xlabel='False positive rate',ylabel='True positive rate');fig.tight_layout();fig.savefig(out/'roc.png',dpi=600);plt.close(fig)
        pre,rec,_=precision_recall_curve(y,p);fig,ax=plt.subplots();ax.plot(rec,pre);ax.set(xlabel='Recall',ylabel='Precision');fig.tight_layout();fig.savefig(out/'precision_recall.png',dpi=600);plt.close(fig)
    fig,ax=plt.subplots();mat=confusion_matrix(ym,np.asarray(pm).argmax(1),labels=[0,1,2,3]);ax.imshow(mat)
    ax.set(xticks=range(4),yticks=range(4),xticklabels=['Fan','Pump','Slider','Valve'],yticklabels=['Fan','Pump','Slider','Valve'],xlabel='Predicted',ylabel='Actual')
    for i in range(4):
        for j in range(4):ax.text(j,i,str(mat[i,j]),ha='center',va='center')
    fig.tight_layout();fig.savefig(out/'machine_confusion.png',dpi=600);plt.close(fig)
