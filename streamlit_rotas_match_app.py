import json
import os
import re
from typing import Dict, Any, List, Tuple, Set

import streamlit as st

# --- defensive import to aid debugging on Streamlit Cloud ---
try:
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from unidecode import unidecode
except ModuleNotFoundError as e:
    st.error(
        "Dependências ausentes. Verifique se o requirements.txt está na RAIZ do repo e contém: "
        "streamlit, scikit-learn, pandas, numpy, unidecode. Depois limpe o cache no Manage app. "
        f"Módulo não encontrado: {e}"
    )
    st.stop()

# =============================
# Helpers • Linguagem
# =============================

PT_STOP_WORDS = [
    'a','à','às','ao','aos','as','o','os','um','uma','umas','uns',
    'de','do','da','dos','das','d','em','no','na','nos','nas','num','numa',
    'por','para','pra','pras','pro','pros','com','sem','sob','sobre','entre',
    'e','ou','mas','tambem','também','como','se','que','quem','quando','onde',
    'porque','porquê','por que','pois','ja','já','mais','menos','muito','muita','muitos','muitas',
    'ser','estar','ter','haver','ir','vai','foi','era','sao','são','serao','serão','seria','seriam',
    'eu','tu','ele','ela','nos','nós','vos','vós','eles','elas','me','te','se','lhe','lhes','nosso','nossa','nossos','nossas',
    'isso','isto','aquilo','este','esta','esse','essa','aquele','aquela','estes','estas','esses','essas','aqueles','aquelas'
]

SENIORITY_ORDER = {
    'estagio': 0, 'trainee': 0,
    'junior': 1, 'júnior': 1, 'jr': 1,
    'pleno': 2, 'mid': 2,
    'senior': 3, 'sênior': 3, 'sr': 3,
    'especialista': 4, 'lead': 4, 'lider': 4, 'líder': 4,
    'gerente': 5,
}

LANG_LEVEL_ORDER = {
    'nenhum': 0,
    'basico': 1, 'básico': 1,
    'intermediario': 2, 'intermediário': 2,
    'avancado': 3, 'avançado': 3,
    'fluente': 4,
    'nativo': 5
}

CONTRACT_TYPES = ['CLT', 'CLT Full', 'PJ', 'Estágio', 'Temporário', 'Tempo Parcial', 'Freelancer', 'Cooperado']

# Léxico simples (pode expandir conforme contexto)
SKILL_SYNONYMS: Dict[str, Set[str]] = {
    'aws': {'amazon web services', 'ec2', 's3', 'rds', 'lambda', 'cloudwatch'},
    'sap basis': {'sap', 'basis'},
    'sql': {'mysql', 'postgres', 'postgresql', 'sql server', 't-sql', 'tsql'},
    'oracle': {'pl/sql', 'plsql', 'oracle database'},
    'itil': {'gestao de incidentes', 'gestão de incidentes', 'mudancas', 'changes', 'major incident'},
    'linux': {'redhat', 'rhel', 'ubuntu', 'debian'},
    'windows': {'active directory', 'ad'},
    'cloud': {'aws', 'azure', 'gcp', 'google cloud', 'cloudops'},
    'gestao': {'lideranca', 'liderança', 'coordenacao', 'coordenação', 'people management'},
}

# =============================
# Utils
# =============================

def norm_text(x: str) -> str:
    if not isinstance(x, str):
        return ''
    return unidecode(x.lower().strip())


def tokenize(text: str) -> List[str]:
    t = norm_text(text)
    toks = re.split(r"[^a-z0-9+]+", t)
    toks = [w for w in toks if len(w) >= 2 and w not in PT_STOP_WORDS]
    return toks


def expand_query_skills(tokens: List[str]) -> Set[str]:
    base = set(tokens)
    expanded = set(base)
    for canon, syns in SKILL_SYNONYMS.items():
        if canon in base or base.intersection({norm_text(s) for s in syns}):
            expanded.add(canon)
            expanded.update({norm_text(s) for s in syns})
    return expanded


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def extract_lang_level(text: str) -> int:
    t = norm_text(text)
    for k, v in LANG_LEVEL_ORDER.items():
        if k in t:
            return v
    if t in {'a2', 'basic'}: return 1
    if t in {'b1', 'b2', 'intermediate'}: return 2
    if t in {'c1', 'advanced'}: return 3
    if t in {'c2', 'fluent', 'native'}: return 4
    return 0


def extract_seniority(text: str) -> int:
    t = norm_text(text)
    for k, v in SENIORITY_ORDER.items():
        if k in t:
            return v
    return 0

# =============================
# Jobs handling
# =============================

def jobs_json_to_df(jobs: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for job_id, job in jobs.items():
        info = job.get('informacoes_basicas', {})
        perfil = job.get('perfil_vaga', {})

        titulo = info.get('titulo_vaga', '')
        area = perfil.get('areas_atuacao', '')
        atividades = perfil.get('principais_atividades', '')
        competencias = perfil.get('competencia_tecnicas_e_comportamentais', '')
        cidade = perfil.get('cidade', '')
        estado = perfil.get('estado', '')
        pais = perfil.get('pais', '')
        nivel_prof = perfil.get('nivel profissional', perfil.get('nivel_profissional', ''))
        nivel_ingles = perfil.get('nivel_ingles', '')
        nivel_espanhol = perfil.get('nivel_espanhol', '')
        contrato = info.get('tipo_contratacao', '')

        # campo textual com pesos (repete título/área para priorizar)
        bag_text = ' \n '.join([
            (str(titulo) + ' ') * 3,
            (str(area) + ' ') * 2,
            str(atividades),
            str(competencias)
        ])

        rows.append({
            'job_id': job_id,
            'titulo': titulo,
            'area': area,
            'atividades': atividades,
            'competencias': competencias,
            'cidade': cidade,
            'estado': estado,
            'pais': pais,
            'nivel_prof': nivel_prof,
            'ingles_req': nivel_ingles,
            'espanhol_req': nivel_espanhol,
            'contrato': contrato,
            'bag_text': bag_text
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_jobs(json_file: str) -> pd.DataFrame:
    with open(json_file, 'r', encoding='utf-8') as f:
        jobs = json.load(f)
    return jobs_json_to_df(jobs)

# Persistência simples: base de candidatos

def save_candidate_record(record: Dict[str, Any], path: str = 'data/candidatos.csv') -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_row = pd.DataFrame([record])
    if os.path.exists(path):
        df_row.to_csv(path, mode='a', header=False, index=False)
    else:
        df_row.to_csv(path, index=False)
    return path

# =============================
# Vectorizer / Scoring
# =============================

@st.cache_resource(show_spinner=False)
def build_vectorizer(corpus: List[str]) -> Tuple[TfidfVectorizer, Any]:
    vect = TfidfVectorizer(
        stop_words=PT_STOP_WORDS,
        min_df=1,
        max_df=0.97,
        ngram_range=(1, 3),
        sublinear_tf=True,
        smooth_idf=True,
        norm='l2'
    )
    X = vect.fit_transform([norm_text(c) for c in corpus])
    return vect, X


def text_similarity_score(vect: TfidfVectorizer, X_jobs, candidate_text: str) -> np.ndarray:
    q = vect.transform([norm_text(candidate_text)])
    sims = cosine_similarity(q, X_jobs)[0]
    # reescala para [0,1] usando percentil 95 (evita dominância de outliers)
    p95 = np.percentile(sims, 95) if sims.size else 1.0
    denom = p95 if p95 > 0 else (np.max(sims) if np.max(sims) > 0 else 1)
    sims = np.clip(sims / denom, 0, 1)
    return sims


def level_match_score(req: str, got: str) -> float:
    if not req:
        return 1.0
    r = extract_lang_level(req)
    g = extract_lang_level(got)
    if r <= 1:  # requisito baixo → não penaliza muito
        return 1.0 if g >= r else 0.7
    return min(1.0, g / max(1, r))


def seniority_match_score(req: str, got: str) -> float:
    r = extract_seniority(req)
    g = extract_seniority(got)
    if r == 0:
        return 1.0
    if g >= r:
        return 1.0
    # defasagem de 1 nível = 0.75, >1 níveis = 0.5
    return 0.75 if r - g == 1 else 0.5


def location_match_score(row: pd.Series, cand_estado: str, cand_cidade: str) -> float:
    if not row.get('estado') and not row.get('cidade'):
        return 1.0
    s = 0.0
    if cand_estado and norm_text(cand_estado) == norm_text(str(row.get('estado', ''))):
        s += 0.6
    if cand_cidade and norm_text(cand_cidade) == norm_text(str(row.get('cidade', ''))):
        s += 0.4
    return s if s > 0 else 0.2


def contract_match_score(req: str, got: str) -> float:
    if not req:
        return 1.0
    return 1.0 if norm_text(req) in norm_text(got) or norm_text(got) in norm_text(req) else 0.6


def skill_overlap_score(job_text: str, cand_tokens: Set[str]) -> float:
    job_tokens = set(tokenize(job_text))
    return jaccard(job_tokens, cand_tokens)


def compute_scores(df: pd.DataFrame, vect: TfidfVectorizer, X_jobs, *,
                   cand_text: str,
                   cand_senior: str,
                   cand_ingles: str,
                   cand_espanhol: str,
                   cand_estado: str,
                   cand_cidade: str,
                   cand_contrato: str,
                   weights: Dict[str, float],
                   text_boost: float = 0.65,
                   skill_boost: float = 0.35) -> pd.DataFrame:

    # Texto (tf-idf) + Overlap de habilidades (jaccard)
    sims = text_similarity_score(vect, X_jobs, cand_text)

    cand_tokens = expand_query_skills(tokenize(cand_text))
    overlap = df['bag_text'].fillna('').apply(lambda t: skill_overlap_score(t, cand_tokens)).values

    # Combina dois sinais textuais
    text_signal = np.clip(text_boost * sims + skill_boost * overlap, 0, 1)

    s_sen = df.apply(lambda r: seniority_match_score(r['nivel_prof'], cand_senior), axis=1).values
    s_en  = df.apply(lambda r: level_match_score(r['ingles_req'],   cand_ingles),  axis=1).values
    s_es  = df.apply(lambda r: level_match_score(r['espanhol_req'], cand_espanhol),axis=1).values
    s_loc = df.apply(lambda r: location_match_score(r, cand_estado, cand_cidade), axis=1).values
    s_ctr = df.apply(lambda r: contract_match_score(r['contrato'],  cand_contrato),axis=1).values

    # Normalização
    s_loc = np.clip(s_loc, 0, 1)

    w = weights
    final = (
        w['texto']       * text_signal +
        w['senioridade'] * s_sen      +
        w['ingles']      * s_en       +
        w['espanhol']    * s_es       +
        w['local']       * s_loc      +
        w['contrato']    * s_ctr
    )

    out = df.copy()
    out['score_texto'] = text_signal
    out['score_texto_tfidf'] = sims
    out['score_texto_overlap'] = overlap
    out['score_senioridade'] = s_sen
    out['score_ingles'] = s_en
    out['score_espanhol'] = s_es
    out['score_local'] = s_loc
    out['score_contrato'] = s_ctr
    out['score_final'] = final

    return out.sort_values('score_final', ascending=False)

# =============================
# UI
# =============================

st.set_page_config(page_title='Roteirizador de Entrevistas • Match de Vagas (v2)', layout='wide')
st.title('🎯 Roteirizador de Entrevistas — Match de Vagas (v2)')

with st.sidebar:
    st.header('⚙️ Dados de Vagas')
    st.caption('O arquivo **deve se chamar exatamente** `vagas.json`.')
    upload = st.file_uploader('Envie o arquivo vagas.json (JSON)', type=['json'])
    default_path = 'vagas.json'
    df_jobs = None
    if upload is not None:
        if upload.name != 'vagas.json':
            st.error('O arquivo enviado deve se chamar **vagas.json**. Renomeie e envie novamente.')
            st.stop()
        jobs = json.load(upload)
        df_jobs = jobs_json_to_df(jobs)
    elif os.path.exists(default_path):
        st.caption('Usando vagas.json encontrado no diretório do app.')
        df_jobs = load_jobs(default_path)
    else:
        st.info('Envie o arquivo **vagas.json** no formato esperado.')

    st.divider()
    st.subheader('🔧 Pesos do Match')
    colw1, colw2 = st.columns(2)
    with colw1:
        w_text = st.slider('Peso — Texto (TF-IDF + Overlap)', 0.0, 1.0, 0.55, 0.05)
        w_sen  = st.slider('Peso — Senioridade',               0.0, 1.0, 0.15, 0.05)
        w_loc  = st.slider('Peso — Localização',               0.0, 1.0, 0.10, 0.05)
    with colw2:
        w_en   = st.slider('Peso — Inglês',                    0.0, 1.0, 0.08, 0.02)
        w_es   = st.slider('Peso — Espanhol',                  0.0, 1.0, 0.05, 0.02)
        w_ctr  = st.slider('Peso — Tipo de Contrato',          0.0, 1.0, 0.07, 0.02)

    total_w = w_text + w_sen + w_loc + w_en + w_es + w_ctr
    weights = {
        'texto':       w_text/total_w,
        'senioridade': w_sen/total_w,
        'local':       w_loc/total_w,
        'ingles':      w_en/total_w,
        'espanhol':    w_es/total_w,
        'contrato':    w_ctr/total_w
    }

    st.subheader('🎚️ Limiar de Match')
    match_threshold = st.slider('Pontuação mínima', 0.0, 1.0, 0.35, 0.01)

    st.subheader('🧪 Parâmetros textuais')
    text_boost = st.slider('Peso interno — TF-IDF', 0.0, 1.0, 0.65, 0.05)
    skill_boost = 1.0 - text_boost
    st.caption(f"Overlap de habilidades = {skill_boost:.2f}")

    recalc = st.button('🔁 Recalcular matches')

st.subheader('📝 Formulário do Entrevistador')
with st.form('form_candidato'):
    c1, c2, c3 = st.columns(3)
    with c1:
        cand_nome = st.text_input('Nome do candidato (opcional)')
        cand_estado = st.text_input('Estado (UF) do candidato')
        cand_cidade = st.text_input('Cidade do candidato')
    with c2:
        cand_senior = st.selectbox('Senioridade do candidato', ['Júnior','Pleno','Sênior','Especialista / Líder','Gerente','Outro'], index=1)
        cand_contrato = st.selectbox('Preferência de contrato', CONTRACT_TYPES, index=0)
        _ = st.text_input('Disponibilidade/Inicio (livre)')
    with c3:
        cand_ingles = st.selectbox('Inglês',   ['Nenhum','Básico','Intermediário','Avançado','Fluente','Nativo'], index=2)
        cand_espanhol = st.selectbox('Espanhol',['Nenhum','Básico','Intermediário','Avançado','Fluente','Nativo'], index=0)
        top_k = st.slider('Quantos matches retornar?', 3, 30, 10, 1)

    cand_skills = st.text_area('Resumo técnico do candidato (stack, ferramentas, certificações, experiências relevantes)', height=150, placeholder='Ex.: AWS, SAP BASIS, SQL, Oracle, gestão de incidentes, ITIL, liderança de times, negociação...')
    cand_obj = st.text_area('Objetivo / áreas de interesse (opcional)', height=80)

    submitted = st.form_submit_button('🔎 Buscar matches')

trigger = submitted or recalc
if trigger:
    if df_jobs is None or df_jobs.empty:
        st.error('Nenhum arquivo de vagas carregado. Envie o **vagas.json** no menu lateral.')
        st.stop()

    with st.spinner('Calculando matches...'):
        vect, X_jobs = build_vectorizer(df_jobs['bag_text'].fillna('').tolist())

        cand_text = " \n ".join([
            cand_senior,
            f"Inglês: {cand_ingles}",
            f"Espanhol: {cand_espanhol}",
            cand_skills,
            cand_obj
        ])

        scored = compute_scores(
            df_jobs, vect, X_jobs,
            cand_text=cand_text,
            cand_senior=cand_senior,
            cand_ingles=cand_ingles,
            cand_espanhol=cand_espanhol,
            cand_estado=cand_estado,
            cand_cidade=cand_cidade,
            cand_contrato=cand_contrato,
            weights=weights,
            text_boost=text_boost,
            skill_boost=skill_boost,
        )

        matches = scored[scored['score_final'] >= match_threshold].copy()

    st.caption(
        f"Pesos usados → Texto: {weights['texto']:.2f} • Senioridade: {weights['senioridade']:.2f} • "
        f"Inglês: {weights['ingles']:.2f} • Espanhol: {weights['espanhol']:.2f} • "
        f"Local: {weights['local']:.2f} • Contrato: {weights['contrato']:.2f} | "
        f"TF-IDF interno: {text_boost:.2f} • Overlap: {skill_boost:.2f}"
    )

    if matches.empty:
        rec = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'nome': cand_nome,
            'estado': cand_estado,
            'cidade': cand_cidade,
            'senioridade': cand_senior,
            'ingles': cand_ingles,
            'espanhol': cand_espanhol,
            'contrato': cand_contrato,
            'skills': cand_skills,
            'objetivo': cand_obj,
            'obs': 'Sem vagas compatíveis no momento'
        }
        path_saved = save_candidate_record(rec)
        st.warning('Não há vagas compatíveis com o perfil informado (acima do limiar). Os dados do candidato foram armazenados na **base de candidatos** para futuras oportunidades.')
        st.caption(f'Base de candidatos: {path_saved}')
        st.stop()

    st.success(f"{len(matches)} vagas compatíveis (limiar {match_threshold:.2f}). Exibindo top {top_k}.")

    cols = ['job_id','titulo','area','cidade','estado','nivel_prof','ingles_req','espanhol_req','contrato','score_final']
    st.dataframe(matches[cols].head(top_k).style.format({'score_final': '{:.3f}'}), use_container_width=True)

    st.divider()
    st.subheader('🔎 Detalhamento por vaga (top resultados)')

    import matplotlib.pyplot as plt

    for _, row in matches.head(min(5, len(matches))).iterrows():
        with st.expander(f"{row['job_id']} — {row['titulo']}  •  Score: {row['score_final']:.3f}"):
            cA, cB = st.columns([2,1])
            with cA:
                st.markdown('**Principais atividades**')
                st.write(row.get('atividades', ''))
                st.markdown('**Competências**')
                st.write(row.get('competencias', ''))
            with cB:
                st.markdown('**Match breakdown**')
                st.metric('Texto (comb.)', f"{row['score_texto']:.3f}")
                st.metric('• TF-IDF', f"{row['score_texto_tfidf']:.3f}")
                st.metric('• Overlap', f"{row['score_texto_overlap']:.3f}")
                st.metric('Senioridade', f"{row['score_senioridade']:.3f}")
                st.metric('Inglês', f"{row['score_ingles']:.3f}")
                st.metric('Espanhol', f"{row['score_espanhol']:.3f}")
                st.metric('Localização', f"{row['score_local']:.3f}")
                st.metric('Contrato', f"{row['score_contrato']:.3f}")

            labels = ['Texto','Senioridade','Inglês','Espanhol','Local','Contrato']
            values = [
                float(row['score_texto']),
                float(row['score_senioridade']),
                float(row['score_ingles']),
                float(row['score_espanhol']),
                float(row['score_local']),
                float(row['score_contrato'])
            ]
            fig, ax = plt.subplots()
            ax.bar(labels, values)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Score (0-1)')
            ax.set_title('Decomposição de Score')
            st.pyplot(fig)
else:
    st.info('Preencha o formulário e clique em **Buscar matches** ou use **Recalcular matches** no menu lateral.')
