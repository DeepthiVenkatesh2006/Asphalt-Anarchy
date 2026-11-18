"""Build the consolidated PDF and simple Word guides from actual experiment files."""
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/final'
HANDOFF=ROOT.parent/'output/team_handoff'


def rows(path):
    with (ROOT/path).open() as f:return list(csv.DictReader(f))


def obj(path):return json.loads((ROOT/path).read_text())
def num(value):return float(value)
def f4(value):return f'{float(value):.4f}'
def pct(value):return f'{float(value)*100:.2f}%'
def table(headers,values):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in values])


def report_markdown():
    cv=rows('reports/member2/local_training/cv_comparison.csv')
    test=rows('reports/member2/local_training/student_test_results.csv')
    extra=rows('reports/audit/model_metrics/student_test_metrics.csv')
    direct={r['model_variant']:r for r in rows('reports/audit/model_metrics/direct_continuity_summary.csv')}
    decoder=rows('reports/decoder/student_test/summary.csv')
    selection=obj('artifacts/local_training/selection.json')
    config=obj('artifacts/local_training/model_metadata.json')
    data=obj('reports/member1/preparation_summary.json')
    validation=obj('reports/audit/validation_summary.json')
    names={'logistic':'Logistic Regression','knn':'Distance-weighted KNN','forest':'Random Forest','histboost':'HistGradientBoosting','mlp':'MLP'}
    model_specs={
      'logistic':('Linear/probabilistic','C=1; lbfgs; max_iter=500; seed=42; StandardScaler','Candidate binary log loss with L2 regularisation.','A fast interpretable baseline for geometric and motion clues. Linear decision boundaries can miss interactions between heading, distance and quality flags.'),
      'knn':('Instance-based','k=9; distance weights; KD-tree; StandardScaler; sample seed=42','No gradient-based fitting loss: similarity-weighted neighbour voting.','Similar candidate clues can have similar labels, without assuming a linear boundary. Its stored examples make prediction and model storage more expensive; the fitting budget was capped at 10,000 complete observation groups.'),
      'forest':('Tree ensemble','80 trees; depth=12; leaf minimum=3; sqrt feature sampling; seed=42','Gini-impurity split criterion; averaged tree probabilities.','Captures nonlinear feature interactions and reduces single-tree variance. Tree depth/leaf controls constrain complexity, but probability calibration and inference cost still need checking.'),
      'histboost':('Boosting','120 iterations; learning rate=.08; 31 leaves; early stopping; seed=42','Candidate binary log loss, improved by successive trees.','Histogram-based trees efficiently model nonlinear tabular features. Early stopping limits iterations, but its internal row holdout is less independent than the outer trip-wise validation.'),
      'mlp':('Neural network','Initial layers=(32,16); Adam; ReLU; alpha=.0001; lr=.001; batch=512; cap=35; seed=42','Candidate binary cross-entropy plus L2 regularisation.','Learns flexible combinations of scaled distance, direction and quality clues. It is sensitive to scaling, optimisation settings and fitting budget; an epoch cap is not evidence of convergence.')}
    merged=[]
    for c in cv:
        t=next(r for r in test if r['model']==c['model'] and r['variant']=='baseline')
        merged.append([names[c['model']],f4(c['observation_macro_f1_mean'])+' +/- '+f4(c['observation_macro_f1_std']),pct(t['top1_accuracy']),f4(t['observation_macro_f1']),f4(direct[c['model']+'_baseline']['direct_continuity']),f4(t['candidate_log_loss']),f"{num(t['predict_seconds']):.4f}"])
    b=next(r for r in test if r['model']=='mlp' and r['variant']=='baseline')
    t=next(r for r in test if r['model']=='mlp' and r['variant']=='tuned')
    s=[]
    def add(text):s.append(text.strip())
    add('# PathSnap: technical report')
    add('Team: Asphalt-Anarchy. Date: 18 November 2026. This consolidated report follows the supplied dataset, training, submission and assessment PDFs. All numerical results below come from saved experiments; local verification is distinguished from cross-device demonstration.')
    add('## 1. Problem and system')
    add('Given noisy GPS observations, predict the supplied road segments travelled. The classifier scores nearby candidate roads; a graph-based HMM/Viterbi decoder chooses a coherent sequence. The API performs inference and separate reference evaluation on the server. A browser or Streamlit client displays returned predictions and aggregate metrics.')
    add('Pipeline: raw GPS and road files -> metric projection -> candidate features -> trained classifier -> graph transitions and Viterbi -> expanded road chunks -> server-side evaluation -> API and Leaflet UI.')
    add(table(['Member','Implemented responsibility'],[['1','Data validation, EDA, reference alignment, candidates and ten features'],['2','Five-model comparison, grouped CV, Bayesian tuning, model export'],['3','Road graph, transition model, Viterbi and route reconstruction'],['4','FastAPI, client interfaces, map and deployment integration']]))
    add('The model checksum is '+validation['model_sha256']+'. The trained weights were not changed during this audit. Correctness repairs were validated against synthetic graph/data cases; performance comparisons use student_test, not the evaluator partition.')
    add('## 2.2 Dataset understanding and exploratory analysis')
    add(table(['Partition','Trips','GPS observations','Purpose'],[['train',80,'208,417','Fitting and grouped CV'],['student_test',10,'19,160','Development comparisons'],['evaluator_test',10,'19,674','Final API evaluation only']]))
    add('Every trip has .track (longitude, latitude, seconds), .nodes (longitude, latitude), .arcs (directed from/to node IDs), and a reference .route where supplied. Node and arc IDs are zero-based row positions within a trip. A route entry is a travelled arc, not a GPS observation; the two row counts need not match.')
    add('All 100 supplied trips passed strict input and reference validation; their IDs are disjoint across partitions. The original 90-trip EDA contains zero rejected trips or nonfinite input values. The processing code validates shape, coordinate ranges, strictly increasing timestamps and referenced IDs. It does not silently fill missing values, sort bad timestamps, delete GPS observations or shift coordinates to a background map.')
    add('A per-trip azimuthal equidistant projection supplies metre distances. At most eight complete road segments within 50 m are candidates for each observation. Graph connectors use all supplied roads, not only candidates. Gaps over 60 seconds split motion/decoder segments. Invalid direction and gap-adjacent speed are represented by explicit flags; a zero placeholder does not prove stationary motion.')
    add(table(['Development output','train','student_test'],[['Candidate rows','1,084,832','113,639'],['Eligible fitting-format candidate rows','755,694','76,234'],['Eligible observations for fitting-format rows','143,533','14,203'],['Aligned observations used for primary evaluation','144,443','14,215'],['Observations without candidates','18,230','0']]))
    add('### Features and feature selection')
    add(table(['Feature','Meaning / reason'],[['distance_m','Candidate proximity in metres'],['heading_delta_deg','Directed difference between motion and road bearing'],['projection_ratio','Position along the candidate, from 0 to 1'],['distance_rank','Relative proximity among shortlisted candidates'],['segment_length_m','Road geometry scale'],['speed_mps','Previous metric displacement / elapsed time when valid'],['dt_s','Time since the preceding GPS observation'],['heading_valid','Whether the motion bearing is reliable'],['gap_before','Whether this observation follows a large time gap'],['zero_length_arc','Explicit degenerate-road indicator']]))
    add('These are manually chosen geometric/motion/quality features, not the result of automatic feature elimination. IDs remain join keys. Reference IDs, target, alignment costs, raw projected coordinates and other diagnostic columns are not model inputs. StandardScaler is fitted only for Logistic Regression, KNN and MLP, inside each fitting fold. No learned imputation or feature log transformation is used.')
    add('### EDA interpretation')
    add('The trip-size histogram shows a right-skewed distribution: long trips can dominate row-weighted fitting. Whole-trip CV and per-trip Macro F1 address evaluation dependence, although most models still fit every eligible row. The road-count scatter shows varying candidate-search difficulty, motivating spatial indexing. The timestamp-gap plot spans several orders of magnitude, motivating motion-validity flags and sequence breaks. Only that plot uses a logarithmic y-axis; the stored feature values are unchanged.')
    add('The maximum training time gap is 153,064 s; the largest student-test gap is 210 s. Training trip 00000054 contributes 16,519 of the 18,230 missing-candidate observations, with median nearest-road distance about 84.2 m. The fixed 50 m radius is therefore an explicit coverage limitation, not a guarantee every GPS point can be matched.')
    add('![Actual development EDA plots](reports/member1/dataset_summary.png)')
    add('### Approximate reference labels and coverage')
    add('The reference arc sequence does not provide one road label per GPS point. A separate monotone spatial alignment derives approximate labels, preserving route occurrences and splitting infeasible chains. Ambiguous, distant, restarted and unmatched labels remain in the ledger but are excluded from the primary aligned-label subset. Fitting also excludes candidate misses; evaluation retains them as incorrect predictions. On student_test, primary accuracy/F1 use 14,215 of 19,160 observations (74.19%). These are approximate references, not independently surveyed point truth.')
    add('## 3. Model experimentation and comparison')
    add('The authoritative experiment fits all five algorithm families on 80 training trips, evaluates every outer GroupKFold split, and holds each training trip out once. No observation from a held-out trip enters that fold\'s fitting/scaler data. MLP and HistGradientBoosting use internal row-level early-stopping holdouts within fitting trips; these are distinct from the outer trip-wise validation.')
    add(table(['Model','CV Macro F1 (SD)','Student Top-1','Student Macro F1','Direct continuity','Log loss','Predict seconds'],merged))
    add('Student scores use the same 74.19% aligned-reference coverage. Candidate log loss is categorical loss after positive-score clipping and within-observation normalisation, conditional on the reference being in the candidate set (99.9156% conditional coverage). Direct continuity measures same-road forward motion or directly adjacent chosen road states before connector insertion; it is not the expanded graph-continuity metric used later. Prediction seconds cover all 113,639 student candidates on the measured Mac and exclude feature extraction/decoding. KNN fits 59,961 candidate rows in its budgeted sample; other models fit 755,694.')
    add('![Five-fold model comparison](reports/member2/local_training/model_comparison.png)')
    add('### 3.2 Questions for each model: suitability, configuration, loss and actual results')
    for name in ['logistic','knn','forest','histboost','mlp']:
        family,params,loss,reason=model_specs[name]
        r=next(x for x in test if x['model']==name and x['variant']=='baseline')
        e=next(x for x in extra if x['model']==name and x['variant']=='baseline')
        c=next(x for x in cv if x['model']==name)
        add('### '+names[name]+' ('+family+')')
        add('Suitability and limitations: '+reason)
        add('Initial configuration: '+params+'. Fitting objective: '+loss)
        add(f"Measured student-test Top-1 {pct(r['top1_accuracy'])}, observation Macro F1 {f4(r['observation_macro_f1'])}, macro precision {f4(e['macro_precision'])}, macro recall {f4(e['macro_recall'])}, conditional log loss {f4(r['candidate_log_loss'])}; final fitting {num(r['fit_seconds']):.3f} s, candidate prediction {num(r['predict_seconds']):.4f} s. CV Top-1 {pct(c['top1_accuracy_mean'])}, CV F1 {f4(c['observation_macro_f1_mean'])} with SD {f4(c['observation_macro_f1_std'])}.")
        add('Over/underfitting evidence: CV and student performance are measured, but a higher student score is not proof against overfitting because trips differ in difficulty. A matched training-set point-accuracy curve was not retained for this model, so no unsupported train-versus-test conclusion is made. Complexity controls and grouped validation reduce risk rather than proving its absence.' if name!='mlp' else 'Over/underfitting evidence: the baseline stopped after 26 of 35 allowed epochs; the tuned fit reached 35. Actual training loss and internal validation accuracy histories are retained. The tuned epoch cap limits convergence claims. Internal row-holdout accuracy is not an independent trip-validation score.')
    add('![Actual candidate confusion matrices for all fitted variants](reports/audit/model_metrics/candidate_confusion_matrices.png)')
    add('These confusion matrices use binary candidate targets on eligible derived-label rows at threshold >0.5. They diagnose candidate classification, not per-observation road-ID accuracy. No artificial loss histories were created for algorithms without epoch histories.')
    add('### 3.3 Selection and preliminary experiments')
    add('MLP had the highest recorded mean CV observation Macro F1, but was nearly tied with HistGradientBoosting; this is not evidence of a statistically significant victory. Its rapid prediction and compatibility with candidate probabilities also support deployment. The older reports/member2/model_comparison files contain a preliminary single grouped fold with different configurations and candidate-binary F1; they are retained as historical evidence and are not interchangeable with the final five-fold observation-level table. Preliminary RF used 100 trees, HistGradientBoosting 150 iterations, and MLP layers (64,32) with a 25-epoch cap. No standalone DecisionTreeClassifier or standard GradientBoostingClassifier is claimed as trained in the final run.')
    add('## 4. Bayesian hyperparameter optimisation')
    add('Only the selected MLP family was tuned. Optuna TPE used seed 42, three startup trials, ten completed trials and three group-aware training folds per trial. Search space: architectures 32, 32/16 or 64/32; alpha 1e-5 to .01 (log scale); learning rate .0003 to .005 (log scale); maximum 35 epochs. The objective was mean aligned-label observation Macro F1. The best trial was compared against the baseline on the same five outer folds; the final variant was chosen before student-test labels were loaded. This is model-selection CV, not nested CV.')
    add('Best settings: layers (64,32), alpha 0.0006358358856676254, learning rate 0.0021992715112745254, 35 epochs. Other fitting settings include Adam, ReLU, batch 512 and early stopping. The final model embeds its fitted StandardScaler.')
    add(table(['Metric','Untuned MLP','Tuned MLP'],[['CV Macro F1',f4(selection['baseline_cv']['observation_macro_f1_mean']),f4(selection['tuned_cv']['observation_macro_f1_mean'])],['CV Top-1',pct(selection['baseline_cv']['top1_accuracy_mean']),pct(selection['tuned_cv']['top1_accuracy_mean'])],['Student Top-1',pct(b['top1_accuracy']),pct(t['top1_accuracy'])],['Student Macro F1',f4(b['observation_macro_f1']),f4(t['observation_macro_f1'])],['Student log loss',f4(b['candidate_log_loss']),f4(t['candidate_log_loss'])],['Student prediction seconds',f4(b['predict_seconds']),f4(t['predict_seconds'])]]))
    add('Tuning modestly improved the declared CV F1 objective. CV Top-1 and CV log loss did not improve; student-test Top-1/F1/log loss did. Do not claim every metric improved or that the fit fully converged. bayesian_trials.csv, the SQLite study, fold_results.jsonl and configurations retain the actual search and computational cost.')
    add('![Actual tuned MLP loss and internal validation accuracy](reports/member2/local_training/mlp_tuned_history_details.png)')
    add('## 5. HMM and Viterbi route reconstruction')
    add('Hidden states are candidate directed road segments for each observation. Candidate-to-road keys retain trip_id, obs_idx, candidate_idx and arc_id. Clipped and normalised positive classifier scores act as discriminative emission potentials; they are not claimed to be calibrated generative P(GPS|road) likelihoods.')
    add('Transition potential is exp(-abs(network distance - observed geodesic distance)/beta), beta=50 m. Network distance includes the remaining previous arc, the shortest directed connector, and the travelled part of the current arc. Forward movement on one arc uses the difference in projection fractions. Returning to a smaller fraction requires a real directed return path. Impossible transitions have zero potential; connector searches use a bounded distance cutoff. Transition rows are normalised over current candidates. The initial state prior is uniform.')
    add('Viterbi recurrence: delta[t,j] = log E[t,j] + max_i(delta[t-1,i] + log T[t-1,i,j]). Backpointers recover the globally best path within each continuous chunk. Missing candidates are -1, not invented road 0. Large gaps and impossible chains split/restart chunks. Expanded routes preserve repeated traversals and intermediate connector arcs; no connector bridges a recorded break.')
    add('Decoder parameters remain the documented fixed settings; no evaluator-based tuning was performed. The required development experiment compares classifier-only and HMM assignments on all ten student-test trips with identical reference coverage and the same frozen trained model.')
    add(table(['Variant','Point accuracy','Observation F1','Road-set F1','Sequence similarity','Graph continuity','Mean inference ms'],[[r['variant'],pct(r['point_accuracy']),f4(r['observation_macro_f1']),f4(r['route_set_f1']),f4(r['sequence_similarity']),pct(r['route_continuity']),f"{num(r['mean_inference_ms']):.1f}"] for r in decoder]))
    add('Classifier Top-1 stays 89.07% because classifier scores do not change. Returned HMM point accuracy is a different metric: 89.53%. Observation F1 rises from .7731 to .8098, while ordered sequence similarity rises from .4360 to .5423. Road-set F1 ignores order; ordered similarity uses SequenceMatcher with autojunk disabled to retain frequent road IDs. Graph continuity includes valid connector paths; gaps/missing points remain breaks. Runtime increases substantially, and timings exclude server-side reference evaluation and network transfer.')
    add('The improvement is not uniform: student trip 00000007 drops in point accuracy from 92.66% to 91.80%, while sequence similarity improves from .5525 to .6221. Trip 00000000 also has a small observation-F1 decline. All per-trip results are retained, rather than showing only successful examples.')
    add('![Measured classifier-versus-HMM comparison](reports/decoder/student_test/decoder_comparison.png)')
    add('## 6. Geographical visualisation using Folium')
    add('The mandatory development visualisation is generated with Folium from actual student-test trip 00000007. Red GPS observations, green development reference roads and dark-blue predicted roads are separate toggleable layers. Road segments are drawn independently rather than joining gaps with fabricated straight lines. Reference geometry is included only in this development report map, never in an evaluator response. The interactive HTML and generator are included in the repository.')
    add('![Actual Folium development map: student-test trip 00000007](reports/final/assets/student_test_00000007_folium.png)')
    add('The live client uses native Leaflet with locally bundled JavaScript/CSS, grey supplied roads, green predictions, blue raw GPS and projected first/last matched observation markers. Street tiles are optional contextual geometry and may differ from the supplied graph. The application never shifts predictions cosmetically to match a basemap.')
    add('## 7. API, UI and deployment')
    add('FastAPI exposes /health, /trips, /predict, /evaluate and /map/{trip_id}, with /api/v1 aliases. Evaluation mode lists only the ten evaluator_test trips and freezes final inference controls. Startup verifies model hashes/version, input schemas and server-side references. An empty/invalid evaluation catalogue is not reported as ready. Invalid IDs/radii/partitions are rejected, and missing trips return 404. Reference validation/evaluation is separated from model features; JSON and evaluator maps contain no raw reference route.')
    add('The Streamlit and browser /ui/ clients share this API. Both clear old results before retries/context changes; nullable/subset metrics show N/A and label coverage. Asynchronous browser responses are tied to request generations. Predicted point assignments and expanded route chunks are separate. UI latency is inference time, not end-to-end network latency.')
    add('Verified achievements: clean dependency installation, 91 passing Python regression tests, four passing JavaScript regressions, real model/decoder processing and rendered route maps. Deployment follows the assignment architecture: Laptop 1 serves the trained model/API and Laptop 2 runs the evaluator client. Capture the receiving-server request logs during the final two-laptop demonstration.')
    add('Laptop 1: python scripts/run_demo.py --role api --data-root STUDENT_DATA_ROOT. Laptop 2: python scripts/run_demo.py --role ui --api-url http://LAPTOP1_IP:8000. Join the same network, confirm /health from Laptop 2, then open http://localhost:8501 there. Use Python >=3.12 and requirements-runtime-lock.txt on both machines.')
    add('## 8. Audit, limitations, reproducibility and integrity')
    add('Audit repairs cover stale UI success, omitted evaluator trips, unsafe error HTML, nullable metrics, invalid readiness, misleading configuration, model integrity/export compatibility, return-loop reconstruction, break preservation and order-sensitive scoring. The current code records separate classifier and decoded metrics. See reports/audit for regression outputs, input/model checksums, extra model metrics and the requirement matrix.')
    add('Limitations remain explicit: reference labels are approximate with partial coverage; model-selection CV is not nested; early stopping uses correlated fitting-row holdouts; the tuned MLP reached its epoch cap; candidate coverage can fail at a fixed radius; HMM adds computation and can hurt individual trips; graph paths and basemaps do not prove the true physical road. Cross-device hardware/network evidence is not inferred from unit tests.')
    add('Reproduce data preparation with src.features.build_features on train and student_test using --include-labels, writing fresh directories. Reproduce the five-model/Optuna run with src.models.local_training and a fresh output directory. scripts/compare_decoder.py uses student_test only. scripts/audit_model_metrics.py verifies every saved model reproduces its original primary metrics; a private KNN cache can be regenerated from training data. The final inference bundle contains model_final.joblib, feature_columns.json, model_metadata.json and artifact_manifest.json; scaler preprocessing is embedded.')
    add('AI assistance was used for parts of implementation, testing and documentation. Numerical claims are backed by actual generated files; AI text is not substituted for experimentation. Team members should review and understand these results. No claim of unaided work or independently surveyed reference truth is made.')
    add('Teacher sources: Main Guidelines (2 pages); Dataset Exploration (4 pages); Models and Training Guidelines (6 pages); Deliverables (7 pages); Assessment and Plagiarism Policy (3 pages). Key evidence: reports/member2/local_training; artifacts/local_training; reports/decoder/student_test; reports/audit; reports/final/assets. Source repository: https://github.com/DeepthiVenkatesh2006/Asphalt-Anarchy, integration branch feature/demo-integration-ready.')
    return '\n\n'.join(s)+'\n'


def make_pdf(markdown,path):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether
    styles=getSampleStyleSheet()
    for name in ['Heading1','Heading2']:
        styles[name].keepWithNext=True
    styles.add(ParagraphStyle(name='ReportBody',fontName='Helvetica',fontSize=9.3,leading=13,spaceAfter=7,splitLongWords=True))
    styles.add(ParagraphStyle(name='Caption',fontName='Helvetica-Oblique',fontSize=8,leading=11,spaceAfter=8))
    styles.add(ParagraphStyle(name='Cell',fontName='Helvetica',fontSize=7.5,leading=10,splitLongWords=True))
    def para(text,style='ReportBody'):
        text=html.escape(text)
        text=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',text)
        return Paragraph(text,styles[style])
    story=[];lines=markdown.splitlines();i=0
    while i<len(lines):
        line=lines[i].strip();i+=1
        if not line:continue
        if line.startswith('|'):
            group=[line]
            while i<len(lines) and lines[i].strip().startswith('|'):
                group.append(lines[i].strip());i+=1
            data=[[x.strip() for x in row.strip('|').split('|')] for row in group if not re.fullmatch(r'[|\- :]+',row)]
            cols=len(data[0]);widths=[(A4[0]-88)/cols]*cols
            cells=[[para(value,'Cell') for value in row] for row in data]
            t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4f1ec')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#cad4d0')),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            story.extend([t,Spacer(1,9)]);continue
        match=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
        if match:
            source=ROOT/match.group(2)
            if not source.is_file():raise FileNotFoundError(source)
            image=Image(str(source));scale=min((A4[0]-88)/image.imageWidth,320/image.imageHeight)
            image.drawWidth=image.imageWidth*scale;image.drawHeight=image.imageHeight*scale
            story.extend([image,para(match.group(1),'Caption'),Spacer(1,9)]);continue
        if line.startswith('### '):story.append(para(line[4:],'Heading2'));continue
        if line.startswith('## '):story.append(para(line[3:],'Heading1'));continue
        if line.startswith('# '):story.append(para(line[2:],'Title'));continue
        story.append(para(line))
    def footer(canvas,document):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#53616a'))
        canvas.drawString(44,25,'Asphalt-Anarchy | PathSnap | verified results and explicit limitations')
        canvas.drawRightString(A4[0]-44,25,str(document.page))
    SimpleDocTemplate(str(path),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=40,bottomMargin=42,title='PathSnap consolidated technical report',author='Asphalt-Anarchy').build(story,onFirstPage=footer,onLaterPages=footer)


def make_word_guides():
    from docx import Document
    from docx.shared import Inches,Pt
    HANDOFF.mkdir(parents=True,exist_ok=True)
    d=Document();section=d.sections[0]
    section.page_width=Inches(8.27);section.page_height=Inches(11.69)
    section.top_margin=section.bottom_margin=Inches(.55)
    section.left_margin=section.right_margin=Inches(.65)
    d.styles['Normal'].font.name='Calibri';d.styles['Normal'].font.size=Pt(10.5)
    d.styles['Normal'].paragraph_format.space_after=Pt(4)
    d.styles['Normal'].paragraph_format.line_spacing=1.0
    for name,size in [('Title',20),('Heading 1',14),('Heading 2',11)]:
        d.styles[name].font.size=Pt(size)
        d.styles[name].paragraph_format.space_before=Pt(7)
        d.styles[name].paragraph_format.space_after=Pt(4)
    def h(text):d.add_heading(text,level=1)
    def p(text):d.add_paragraph(text)
    d.add_heading('PathSnap: simple whole-project study guide',0)
    p('Current audited explanation. Older fixed demo percentages and pending-work statements are superseded by this guide. Read the short speech first, then the roles and questions.')
    h('Say this in your presentation')
    p('PathSnap finds the roads a vehicle probably travelled on using noisy GPS readings. I prepare the data and useful clues. Member 2 trains models to score nearby roads. Member 3 uses road connections and Viterbi to choose a sensible route. Member 4 connects everything to an API and shows the result on a map.')
    h('The four dataset files')
    for text in ['Nodes: fixed road points.','Arcs: directed connections between those road points.','Track: GPS locations and recording times from the journey.','Route: the supplied reference road sequence used for training labels or later evaluation, not as a prediction input.'] :p(text)
    p('There are 80 training trips, 10 student-test trips and 10 final evaluator trips: 100 total. A trip ID is a name, not a running count. GPS point rows do not correspond one-to-one to route entries.')
    h('Member 1: your part')
    p('I checked the GPS and road files, validated coordinates/times/IDs, kept every GPS observation and flagged time gaps. I found up to eight possible roads within 50 metres of each GPS point and calculated ten clues for the model. I separately derived approximate reference labels from the route sequence.')
    for text in ['Distance: how far the GPS point is from the road.','Direction difference: whether vehicle motion and road direction agree.','Position along the road: fraction from its start to its end.','Distance rank: nearest candidate is rank 1.','Road length: length of that road segment.','Estimated speed: distance moved divided by time when valid.','Time interval: time between GPS readings.','Direction-valid flag: whether a reliable motion direction is available.','Gap flag: whether the reading follows a gap over 60 seconds.','Zero-length flag: whether the road geometry has identical endpoints.']:p(text)
    p('We did not perform automatic feature elimination or log-transform the features. IDs and reference/target columns are excluded from model inputs. The model receives the ten named features only.')
    h('What to explain in the CSV opened in Excel')
    p('One row is one possible road for one GPS point. The same obs_idx repeats because several candidate roads are considered. obs_idx starts at 0 within each trip. candidate_idx identifies a candidate; arc_id identifies its road within that trip. distance_m and the other ten clues are model features. A development target of 1 means the candidate agrees with the approximately derived reference label; target is not passed during prediction.')
    h('The three graphs: one sentence each')
    p('Trip-size histogram: journeys have different lengths, so long trips can dominate row-based fitting and validation must keep whole trips separate.')
    p('GPS count versus road count: available road networks vary in size, so we use spatial candidate filtering rather than scoring every road.')
    p('Maximum time-gap scatter: long gaps make speed and direction unreliable, so we use flags and break route sequences. Its y-axis is logarithmic for display only.')
    d.add_page_break()
    h('Member 2: models and tuning')
    p('The model learns which candidate roads fit the clues. It outputs a score for each possible road. We trained Logistic Regression, KNN, Random Forest, HistGradientBoosting and MLP. The actual final five do not include a standalone Decision Tree or standard GradientBoostingClassifier.')
    p('Logistic Regression combines clues simply. KNN looks for similar examples. Random Forest combines trees. HistGradientBoosting improves trees in stages. MLP is a small neural network that learns feature combinations.')
    p('Five-fold GroupKFold keeps entire trips together. Scaling is fitted only on the fitting trips. We selected MLP using training CV, then used ten Optuna TPE trials to tune it. Final layers are 64 and 32; the scaler is saved inside the model. Random seed is 42. KNN uses a documented fitting sample for speed.')
    h('Member 3: HMM and Viterbi')
    p('Choosing a road independently at every GPS point can cause jumps. HMM combines how well each road fits the GPS point with how plausible the movement between roads is. Viterbi finds the best sequence using those two kinds of scores.')
    p('Emission score: classifier support for a candidate. Transition score: agreement between observed movement and directed network distance. The graph includes all roads, so intermediate connector roads can be reconstructed. Missing candidates stay unmatched as -1; gaps and impossible transitions split chunks. A road can be visited again through a valid return loop.')
    h('Member 4: API and map')
    p('The server loads GPS and road data, builds features, runs the saved model and decoder, then evaluates against references separately. The client selects a trip and displays the returned predictions and aggregate metrics. Final evaluator mode exposes ten evaluator trips and freezes prediction controls.')
    p('Live maps use Leaflet. Grey means supplied roads, green means predicted roads and blue means raw noisy GPS. First/last markers use matched GPS positions. White background streets may differ from the supplied road dataset. We do not shift coordinates to make the map look correct. Folium is also used for the required development/report map, where reference routes may be shown.')
    h('Numbers to remember, with their limits')
    p('On the ten student-test trips: classifier Top-1 accuracy is 89.07%; HMM returned-point accuracy is 89.53%. Observation Macro F1 improves from 0.7731 to 0.8098. These point scores use 14,215 aligned approximate labels out of 19,160 GPS readings: 74.19% coverage. They are not verified physical ground truth for every point.')
    p('HMM does not improve every trip. Trip 00000007 has slightly worse point accuracy after decoding, although its sequence improves. Decoding also costs more runtime. Do not memorise the old 94.5% / 99.8% demo claims as final project-wide scores.')
    d.add_page_break()
    h('Quick answers for the review')
    for question,answer in [
      ('Why not choose the nearest road?','Noise and parallel roads can make the nearest road wrong; direction and sequence help.'),
      ('What is data cleaning here?','Strict validation and explicit flags, not silently filling missing values or moving GPS points.'),
      ('What is data leakage?','Giving the model information about the answer that would not be available during prediction. We exclude reference/target columns and separate trips in validation.'),
      ('Why are GPS dots beside the green line?','The dots are noisy original recordings; the line represents selected supplied roads.'),
      ('Why can the white road differ?','It comes from a separate map provider; the model matches the supplied road network.'),
      ('Did tuning improve everything?','No. It improved the chosen CV F1 objective, while CV Top-1 and log loss did not improve.'),
      ('What is Macro F1?','It balances precision and recall across road classes; our observation result is calculated per trip then averaged.'),
      ('What is sequence similarity?','An order-sensitive comparison of predicted and reference road sequences. Road-set F1 ignores order and is a separate metric.'),
      ('Is the HTML for all trips?','Each generated map shows a selected trip. The UI lets us choose another trip and request a new prediction.')]:
        d.add_paragraph(question,style='Heading 2');p(answer)
    h('Two-laptop demonstration')
    p('Laptop 1 holds the trained model and dataset. Run: python scripts/run_demo.py --role api --data-root STUDENT_DATA_ROOT. Laptop 2 runs: python scripts/run_demo.py --role ui --api-url http://LAPTOP1_IP:8000. Open http://localhost:8501 on Laptop 2.')
    p('Connect both laptops to the same network. Check the API health page from Laptop 2, select an evaluator trip and run a prediction. Show the Leaflet route and the incoming request in Laptop 1’s terminal.')
    p('Remember: prepare clues -> score roads -> connect the route -> display the result.')
    destination=HANDOFF/'PathSnap_Whole_Project_Simple_Study_Guide.docx';d.save(destination)
    # Replace stale companions while retaining their earlier versions privately.
    archive=HANDOFF/'archive/pre_audit';archive.mkdir(parents=True,exist_ok=True)
    for name in ['PathSnap_Team_Guide.docx','Member1_Presentation_Notes.docx']:
        old=HANDOFF/name
        if old.exists() and not (archive/name).exists():shutil.copy2(old,archive/name)
        companion=Document(destination)
        companion.paragraphs[0].text='PathSnap team execution guide' if name.startswith('PathSnap_Team') else 'Member 1 presentation and project notes'
        if name.startswith('PathSnap_Team'):
            companion.add_page_break();companion.add_heading('Member commands and handoff prompts',level=1)
            commands=[
                ('Member 1', 'python -m src.data.eda_inspector --input-dir "DATA_ROOT" --output-dir reports/member1/reproduced_eda', 'Check loaders, local projections, candidate IDs and ten-feature allowlist. Preserve observations and keep reference alignment separate from prediction.'),
                ('Member 2', 'python -m src.models.local_training --data-root data/processed/v1 --output-dir reports/member2/new_run --folds 5 --tuning-folds 3 --trials 10', 'Compare all five families on training-trip CV, use only permitted development data, and export the model, feature schema, metadata and integrity manifest together.'),
                ('Member 3', 'python scripts/compare_decoder.py --output-dir reports/decoder/new_comparison', 'Preserve candidate mapping and projected positions. Test directed connectors, return loops, gaps and unmatched points; compare fixed-model classifier and HMM results on student_test.'),
                ('Member 4', 'python scripts/run_demo.py --role api --data-root "DATA_ROOT"', 'Use the audited API/schema and pinned runtime. Verify ten evaluator trips, frozen controls, reference separation, mobile/client errors and real receiving-server request logs.')]
            for member,command,prompt in commands:
                companion.add_heading(member,level=2);companion.add_paragraph(command);companion.add_paragraph('Prompt: '+prompt)
            companion.add_heading('GitHub workflow',level=2)
            companion.add_paragraph('Fetch origin and use feature/demo-integration-ready as the shared audited base. Keep personal changes on separate branches, run tests, commit, push and request review. Do not overwrite teammates or commit datasets/environments. Complete source is required, not only model weights.')
        companion.save(old)
    return destination


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    markdown=report_markdown()
    (OUT/'TECHNICAL_REPORT.md').write_text(markdown)
    make_pdf(markdown,OUT/'PathSnap_Technical_Report.pdf')
    guide=make_word_guides()
    print(OUT/'PathSnap_Technical_Report.pdf');print(guide)


if __name__=='__main__':main()
