from xml.sax.saxutils import escape
from collections import Counter
import platform

def footer(canvas, document):
    canvas.saveState()
    canvas.setFont('Arial', 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(15*mm, 8*mm, 'SWEET.TV | Petr Vorozhtsov | Detailed report v2')
    canvas.drawRightString(195*mm, 8*mm, str(document.page))
    canvas.restoreState()

def h(t): story.append(P(t, 'H1Ru'))
def p(t): story.append(P(t))
def table(rows, widths):
    tab=Table([[P(escape(str(c)), 'SmallRu') for c in row] for row in rows], colWidths=widths, repeatRows=1)
    tab.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.3,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6edf4')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.append(tab)

titles={
'2500413':'Civil Defence', '700389':"Who Is on Top? Season 6, Episode 7",
'10002893':'When We Are at Home', '5800509':'Real Mysticism',
'500281':'This Is Ukraine Speaking', '90079323':'Now You See Me 2',
'90077525':'Plastic', '90072921':'Big Soldier', '90078430':'Kingsman: The Secret Service',
'90079464':'Deadpool', '10003159':'Nothing Happens Twice, Episodes 1-2',
'700390':'Superintuition, Season 3, Episode 4: Andrey Bednyakov and Yulia Sanina',
'2400508':'Today: Results with Oleg Panyuta', '20088':'Breakfast with 1+1',
'200337':'Morning with Inter', '6200371':'Jedi', '400335':'A House for Dad, Season 7, Episode 1',
'500368':'60 Minutes with Olga Skabeyeva and Evgeny Popov', '90082757':'Bernie the Dolphin',
'90082758':'Bernie the Dolphin 2', '90076672':'Courier from Paradise',
'90081355':'The Extraordinary Journey of the Fakir',
'12002955':'Peppa Pig', '2400467':'Window News', '2400480':'TSN: Week',
'400426':'Wife Swap', '240081':'Facts: Day',
}
categories={'Фильмы':'Films','Развлечения':'Entertainment','Познавательное':'Educational',
'Детям':'Children','Инфо':'Information','Сериалы':'Series','Другое':'Other',
'Спорт':'Sport','Музыка':'Music'}
def cat_en(sid): return categories.get(meta.get(sid, ('',''))[1], 'Unspecified')
def title_en(sid): return titles.get(sid, 'Programme '+sid)

story=[]
story += [P('SWEET.TV: TV Programme Recommendation', 'TitleRu'), P('Detailed Report v2', 'TitleRu'), P('Petr Vorozhtsov | Kaggle: PetrVorozhtsovLMS')]
p('Implementation and final submission: <link href="https://github.com/PetrVorozhtsov/RecSys-LLMs/tree/exam-sweettv/exam_sweettv" color="blue">GitHub: RecSys-LLMs / exam-sweettv / exam_sweettv</link>.')
h('1. Task and results')
p('The task is to produce a ranked list of five programme IDs for each of 1,260 users for future weeks 31-42. Two personalised methods were implemented: content-based filtering and association scoring. Popularity provides the required reference and a fallback list.')
table([['Result','Value'],['Submitted method','Association: co-view confidence and lift-like weighting'],['Local MAP@5: association','0.157865'],['Local MAP@5: popularity','0.162263'],['Local MAP@5: content','0.008688'],['Kaggle Private Score','0.01454'],['Kaggle Public Score','0.01740'],['Prediction file','outputs/submission.csv: 1,260 rows, 5 IDs per row']], [65*mm,110*mm])
h('Main finding')
p('Association outperformed content among the two implemented personalised methods. However, popularity exceeded association by 0.004398 MAP@5, approximately 2.71% of the baseline score. These results therefore do not establish an advantage for personalisation. The submitted Kaggle solution uses association; the screenshot on the final page confirms its returned private score.')
p('The final selector in the implementation compares content with association, rather than selecting the best of all three methods. The experimental results and submitted CSV were not changed when preparing this expanded report.')
h('2. Sources and scope of evidence')
p('Sources are the supplied assignment, competition CSV files, run.py, and the saved audit.json, validation_results.json and examples.json. The student supplied the Kaggle screenshot. The Codex AI assistant helped implement the solution and prepare the report. run.py uses no external pretrained models or additional user data. No online experiments or revenue effects were measured.')
story.append(PageBreak())
h('3. Data preparation and interaction construction')
p('dataset11-30.csv contains 5,821,987 viewing intervals. The user key is user_id; vsetv_id is matched to schedule channel_id. The misspelled duraton field describes the viewing interval, not programme length. The implementation obtains viewing boundaries directly from start_time and stop_time. The schedule is divided into weeks 11-20, 21-30 and 31-42.')
p('Schedule rows with an empty ID or tv_show_id = 0 are excluded because they do not identify recommendable items. According to the assignment, only approximately 10.8% of schedule rows carry a nonzero programme ID. This describes the source schedule, not the successful join rate in this run.')
h('Interval matching rule')
p('For the same channel, overlap = min(stop_view, stop_show) - max(start_view, start_show). A match requires overlap of at least 60 seconds and overlap / duration_show of at least 0.10. If several labelled programmes overlap, the single eligible programme with maximum overlap is selected. Repeated matches to the same ID accumulate overlap seconds as the user-item weight.')
p('The 60-second threshold removes very short channel switches; the 10% threshold scales the condition to programme length. These are engineering choices, not tuned optima: no sensitivity study was performed. With the 396-second median viewing interval stated in the assignment, the absolute cutoff does not automatically reject a typical view. Nevertheless, 10% may be too strict for long programmes, while selecting a single match loses information from long viewing sessions.')
table([['Join audit','Count'],['Input viewing intervals',f"{join_audit['views']:,}"],['Excluded: users outside the submission template',f"{join_audit['filtered_users']:,}"],['No eligible labelled match',f"{join_audit['unmatched_or_unlabelled']:,}"],['Matched viewing intervals',f"{join_audit['matched']:,}"],['Users with weeks 11-20 history',audit['users_train']],['Users with combined history',audit['users_final']],['Final catalogue size',audit['candidate_final']]], [120*mm,55*mm])
p('Restricting the log to template users leaves 2,173,313 intervals; the 502,213 matched intervals represent approximately 23.11% of this subset. The matched counter measures intervals, not distinct user-item pairs. Co-view and popularity training are also restricted to template users, reducing the available collaborative signal.')
story.append(PageBreak())
h('4. Validation protocol and candidate pool')
p('Training uses viewing behaviour from weeks 11-20, while relevant validation IDs come from weeks 21-30. This preserves temporal order relative to future weeks 31-42. Validation candidates are the union of nonzero IDs in schedules 11-20 and 21-30. The final pool is the union of all three schedules, containing 9,671 IDs; it is not restricted to programmes airing during the test period.')
p('Already-watched IDs are excluded identically across methods. This is a product choice, although the competition permits repeat viewing. Validation relevance sets still include repeat views; some relevant items are therefore deliberately unreachable for every method.')
h('MAP@5 definition')
p('For each user, Precision@r is summed over ranks 1 through 5 where the recommended ID belongs to the relevant set. This sum is divided by min(5, number of relevant IDs). MAP@5 averages AP@5 over users with a nonempty validation history. Ordering matters because earlier hits contribute more. Scores from different models are not probabilities.')
p('Implementation caveat: validation predictions are generated only for users present in training. A user appearing only in validation remains in the metric denominator but receives an empty prediction and a zero contribution. Thus the fallback described below is not fully applied to cold-start users in the local evaluation. Their count was not saved as a separate audit field.')
h('5. Methods and parameters')
table([['Method','Ranking and parameters'],['Popularity','Number of distinct training viewers per ID; ordering uses Counter.most_common().'],['Content','Category weight: 2; genre: 1; production decade: 0.5; director and actor tokens longer than 2 characters: 0.25. User profile: sum of item vectors weighted by log(1 + overlap_seconds). Cosine similarity; ties use the string ID.'],['Association','Co-occurrence of IDs in user histories. Pair construction uses the first 60 IDs per history in insertion order, not necessarily the latest or most important programmes.'],['Top-5 / fallback','The method ranking is followed by popularity. Watched IDs and duplicates within the recommendation list are removed.']], [34*mm,141*mm])
p('Association confidence(a,b) = pair_count(a,b) / item_users(a). The implemented lift term is confidence / (item_users(b) / sum(item_users.values())). The denominator uses total item support rather than the number of users. This is a lift-like adjustment, not classical user-probability lift. The final contribution is log(1 + overlap_user_a) * confidence * log(1 + lift).')
story.append(PageBreak())
h('6. Interpretation and implementation limitations')
p('Popularity achieves the highest local MAP@5. Association is close but lower, while content is substantially weaker. This is consistent with strong overall popularity and limited metadata, but the causal contribution of each factor has not been established: no ablation, confidence intervals or user-group evaluations were calculated.')
p('Final user profiles and fallback popularity use weeks 11-30. However, the association pair table remains fitted on weeks 11-20: the code does not rebuild pairs after combining histories. Items first appearing in weeks 21-30 consequently gain no new co-view links. This is a limitation of the actual final model, which is not fully refitted on the expanded window.')
p('Metadata from all schedules is merged before validation; later schedule metadata can replace descriptions for repeated IDs. Future viewing behaviour is not used. Nevertheless, using attributes from a future schedule goes beyond a strict availability-only interpretation of test schedule access. A stricter comparison should freeze metadata at the training cutoff and reevaluate content. The reported numbers refer to the current implementation.')
h('Strategy for the platform')
p('For the next controlled iteration, popularity is the justified starting point: it is the strongest measured method, inexpensive to compute, and provides a clear cold-start fallback. Association remains a personalisation candidate after correcting lift normalisation, broadening the training audience, and fully refitting on weeks 11-30. The association file already submitted to Kaggle remains the designated final submission for this work; no private-score comparison with popularity was performed.')
p('Content can potentially generate candidates for new programmes without co-view evidence. Its current measured quality is too low to justify deployment. Field weights and name normalisation need investigation. Rank fusion and content reranking are possible future experiments, not implemented and validated improvements.')
h('Computational properties and reproducibility')
p('Schedules are stored by channel with sorted start times; bisect_right bounds candidate lookup during the join. Content scans the catalogue per user. Association generates at most 60 * 59 / 2 = 1,770 unordered pairs per user before aggregation. Popularity is linear in the number of distinct user-item links. Actual peak memory and total runtime were not recorded; no numerical runtime or memory claim is made.')
p(f'Report generation environment: Windows, Python {platform.python_version()}, ReportLab for PDF generation and Poppler for rendering checks. run.py uses the Python standard library, no neural-network training, and no random seed. Ordering of tied counters depends on input order; exact reproduction requires retaining source files and their order.')
p('From exam_sweettv, run: python run.py --data-dir data --out-dir outputs. The JSON outputs preserve the audit, metrics and two examples. To regenerate this PDF, run python make_report_v2.py with report_v2_english.py, the CSV files, and outputs/score-screen-shot.png available.')
story.append(PageBreak())
h('7. Comparable examples for two users')
p('The first two template users were selected and saved in examples.json. This choice is reproducible, but it is neither a representative sample nor a heavy-versus-cold-start comparison: both users have history. The same users are used across all methods. Their histories combine weeks 11-30; actual future viewing in weeks 31-42 is unavailable. Programme titles below are English display translations; IDs retain the exact source identity.')
for user,data in examples.items():
    h(f'User {user}: {len(data["history"])} historical IDs')
    cats=Counter(cat_en(sid) for sid in data['history'])
    p('Distinct watched programmes by category: '+escape('; '.join(f'{k}: {v}' for k,v in cats.most_common()))+'. These are ID counts, not viewing-time shares.')
    p('Observed history examples: '+escape('; '.join(f'{sid}: {title_en(sid)} ({cat_en(sid)})' for sid in data['history'][:4]))+'.')
    rows=[['Method','Ordered Top-5: programme ID, English title and category']]
    for method,preds in data['predictions'].items():
        rows.append([method,'; '.join(f'{n}. {sid}: {title_en(sid)} ({cat_en(sid)})' for n,sid in enumerate(preds[:5],1))])
    table(rows,[25*mm,150*mm])
    a=set(data['predictions']['association']); b=set(data['predictions']['popularity'])
    p(f'Association and popularity share {len(a & b)} of 5 IDs. All displayed recommendations are absent from the saved history by ID because of the filtering rule. This is not evidence of useful discovery or of no previous viewing outside this platform.')
p('For the first user, content mostly recommends films, whereas association and popularity include shows and series. For the second user, association preserves the popular item set with a different order. These observations should be read alongside the historical category counts above. Individual hits in the future test period cannot be verified from the supplied viewing log.')
story.append(PageBreak())
h('8. Kaggle evidence and submission package')
p('The student uploaded submission.csv and supplied the screenshot below. Status: Complete (after deadline). Private Score = 0.01454; Public Score = 0.01740. These Kaggle results should not be equated directly with local MAP@5 because the users, time windows, candidate pool and target construction differ.')
img=Image(str(ROOT/'outputs/score-screen-shot.png')); img._restrictSize(176*mm,120*mm); story.append(img)
p('Figure 1. The supplied Submissions screenshot showing the returned private and public scores for the designated final submission.csv.')
h('Files and course submission')
table([['Artifact','Purpose'],['run.py','Join, models, validation and submission generation'],['outputs/submission.csv','Final file uploaded to Kaggle by the student'],['outputs/audit.json','Processing counters and catalogue sizes'],['outputs/validation_results.json','MAP@5 and the selected method'],['outputs/examples.json','The same two users for all three methods'],['outputs/score-screen-shot.png','Evidence of returned Kaggle scores'],['report_v2.pdf','This detailed English report; report.pdf is preserved']], [70*mm,105*mm])
p('The course submission requires the PDF and the original, unmodified AI-session export. The session export is a separate file and is not embedded in this PDF. The CSV and implementation belong in the repository. This version claims no new Kaggle submission or additional quality measurements.')
h('Checks for the next iteration')
p('Restrict the final pool to the relevant airing schedule; explicitly evaluate repeat viewing; fix cold-start validation; use the correct user denominator for lift; train co-view on a wider audience; isolate metadata by time; and fully refit the final model. Each change requires fresh temporal validation against popularity before replacing the submitted solution.')
