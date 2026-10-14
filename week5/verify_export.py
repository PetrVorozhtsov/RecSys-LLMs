"""Recalculate a browser-exported model using NumPy, without project JS/TF.js."""
from pathlib import Path
import argparse, collections, json, math
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('run',type=Path)
parser.add_argument('--repo',type=Path,default=Path('work/RecSys-LLMs/week5'))
parser.add_argument('--out',type=Path,default=Path('work/independent-model-check.json'))
args=parser.parse_args()
run=json.loads(args.run.read_text(encoding='utf-8'))
allratings=[]
for line in (args.repo/'u.data').read_text(encoding='ascii').splitlines():
    u,i,r,t=map(int,line.split('\t'));allratings.append((u,i,r,t))
genres=["Action","Adventure","Animation","Children's","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"]
movies={}
for line in (args.repo/'u.item').read_text(encoding='latin-1').splitlines():
    fields=line.split('|');movies[int(fields[0])]={'title':fields[1],'genres':[g for g,f in zip(genres,fields[6:24]) if int(f)]}
byuser=collections.defaultdict(list)
for row in allratings:byuser[row[0]].append(row)
train=[];val=[];test=[]
for u,rows in byuser.items():
    rows.sort(key=lambda row:(row[3],row[1]))
    a=int(len(rows)*.8);b=int(len(rows)*.1)
    train.extend(rows[:a]);val.extend(rows[a:a+b]);test.extend(rows[a+b:])
users=set(row[0] for row in train);items=set(row[1] for row in train)
train_support=collections.Counter(row[1] for row in train)
fm=run['factorMatrices']
p=np.array(fm['userWeights'],dtype=np.float64).reshape(fm['userShape'])
q=np.array(fm['itemWeights'],dtype=np.float64).reshape(fm['itemShape'])
mu=sum(r[2] for r in train)/len(train)
assert fm['factorCount']==16 and set(fm['trainMovieIds'])==items and set(fm['trainUserIds'])==users
metrics={}
for name,rows in [('train',train),('validation',val),('test',test)]:
    eligible=[row for row in rows if row[0] in users and row[1] in items]
    u=np.array([row[0] for row in eligible]);i=np.array([row[1] for row in eligible]);actual=np.array([row[2] for row in eligible])
    prediction=np.sum(p[u]*q[i],axis=1)
    rmse=float(np.sqrt(np.mean((prediction-actual)**2)))
    baseline=float(np.sqrt(np.mean((mu-actual)**2)))
    sourcekey={'train':'selectedTrainRMSE','validation':'selectedValRMSE','test':'selectedTestRMSE'}[name]
    app=run['metrics'][sourcekey]
    assert abs(app-rmse)<2e-6,(name,app,rmse)
    metrics[name]={'rawCount':len(rows),'eligibleCount':len(eligible),'excluded':len(rows)-len(eligible),'numpyMF_RMSE':rmse,'browserMF_RMSE':app,'absoluteDifference':abs(app-rmse),'baselineRMSE':baseline,'squaredErrorSum':float(np.sum((prediction-actual)**2))}
assert abs(run['metrics']['baselineTestRMSE']-metrics['test']['baselineRMSE'])<1e-12
selected=min(run['history'],key=lambda x:(x['valRMSE'],x['epoch']))
assert selected['epoch']==run['selectedEpoch']
assert abs(selected['valRMSE']-metrics['validation']['numpyMF_RMSE'])<2e-6
past=collections.defaultdict(set)
for row in train+val:past[row[0]].add(row[1])
pool=np.array(sorted(items))
scores=q[pool]@p.T
top={}
for u in users:
    candidates=[(int(i),float(s)) for i,s in zip(pool,scores[:,u]) if int(i) not in past[u]]
    candidates.sort(key=lambda x:(-x[1],x[0]));top[u]=candidates[:10]
relevant=collections.defaultdict(set);unknownrelevant=set();testusers=set()
for u,i,r,t in test:
    testusers.add(u)
    if r>=4:
        if i not in items:unknownrelevant.add(i)
        elif u in users and i not in past[u]:relevant[u].add(i)
peruser=[]
for u,films in sorted(relevant.items()):
    hit=len(films.intersection(i for i,s in top[u]));peruser.append({'userId':u,'relevant':len(films),'hit':hit})
recall=sum(row['hit']/row['relevant'] for row in peruser)/len(peruser)
originalrecall=run['metrics']['recallAt10']
assert abs(recall-originalrecall['overallRecall'])<1e-12
assert peruser==originalrecall['perUser']
assert unknownrelevant==set(originalrecall['excluded']['unseenIds'])
assert testusers-set(relevant)==set(originalrecall['excluded']['usersWithoutRelevant'])
recommendations={};profiles={}
for u in [1,2]:
    rec=run['recommendations']['user'+str(u)]
    assert [x['movieId'] for x in rec]==[i for i,s in top[u]]
    rows=[]
    historical=[row for row in train+val if row[0]==u]
    oldgenres=set(g for row in historical for g in movies[row[1]]['genres'])
    recgenres=collections.Counter()
    for rank,(i,s) in enumerate(top[u],1):
        assert i not in past[u]
        movie=movies[i];newgenres=[g for g in movie['genres'] if g not in oldgenres]
        recgenres.update(movie['genres'])
        rows.append({'rank':rank,'movieId':i,'title':movie['title'],'score':s,'browserScore':rec[rank-1]['score'],'genres':movie['genres'],'newGenres':newgenres,'trainingRatingCount':train_support[i]})
    recommendations[str(u)]=rows
    profiles[str(u)]={'pastRatings':len(historical),'pastGenres':sorted(oldgenres),'recommendedGenreCounts':dict(recgenres),'newTitleFraction':1.0,'recommendedMoviesWithNewGenre':sum(bool(row['newGenres']) for row in rows),'testRelevantEligible':len(relevant.get(u,set())),'hits':next((r['hit'] for r in peruser if r['userId']==u),0)}
audit=run['independentPredictionAudit']
dot=float(np.dot(p[audit['userId']],q[audit['movieId']]))
assert abs(dot-audit['tensorPrediction'])<2e-6
payload={'method':'Independent Python/NumPy float64 parsing, splitting, scoring and full-pool ranking; no application JavaScript or TensorFlow imported.','selectedEpoch':run['selectedEpoch'],'trainMean':mu,'metrics':metrics,'recallAt10':{'numpy':recall,'browser':originalrecall['overallRecall'],'users':len(peruser),'totalHits':sum(x['hit'] for x in peruser),'totalEligibleRelevant':sum(x['relevant'] for x in peruser),'unseenRelevantMovies':len(unknownrelevant),'excludedUsers':len(testusers-set(relevant))},'predictionAudit':{'userId':audit['userId'],'movieId':audit['movieId'],'actual':audit['actual'],'numpyDot':dot,'browser':audit['tensorPrediction'],'absoluteDifference':abs(dot-audit['tensorPrediction']),'squaredError':(dot-audit['actual'])**2},'recommendations':recommendations,'profiles':profiles,'passed':True}
args.out.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in payload.items() if k not in ['recommendations','profiles']},indent=2))
