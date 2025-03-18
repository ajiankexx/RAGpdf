# TODO: raptor核心逻辑
from base import llm_chat, llm_embedding
import umap
from sklearn.mixture import GaussianMixture
import trio
import numpy as np
import logging

class Raptor():
    def __init__(self, config):
        self._threshold = config.raptor_threshold
        self._max_cluster = config.raptor_max_cluster
        self._prompt = config.raptor_prompt

    def _get_n_cluster(self, embeddings, random_state: int):
        max_cluster = min(len(embeddings), self._max_cluster)
        all_possible_clusters = [i+1 for i in range(max_cluster)]
        bics = []
        for n in all_possible_clusters:
            gm = GaussianMixture(n_components=n, random_state=random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        optimal_cluster = all_possible_clusters[np.argmin(bics)]
        return optimal_cluster

    def _summarize(self, chunks, idx: list):
        _text = [chunks[i][0] for i in idx]
        cluster_content = "\n".join(i for i in _text)
        messages = [
            {"role":"system", "content":"You are a helpful assistant."},
            {"role":"user", "content":self._prompt.format(cluster_content = cluster_content)}
        ]
        text = llm_chat(messages)
        logging.debug(f"cluster_content: {text}")
        embed = llm_embedding(text)
        chunks.append((text, embed))
        return chunks


    def build_tree(self, chunks, random_state):
        """
        chunks:[(text, embedding)]
        """ 
        optimal_cluster = self._get_n_cluster(chunks, max_cluster)

        start = 0
        end = len(chunks)
        label = [i for i in range(len(chunks))]
        layer = [(0, len(chunks))]
       
        while end - start > 1:
            if end - start == 2:
                label.extend([0, 0])
                self._summarize(chunks, [start, start + 1])
                start = end
                end = end + 1
                layer.append((start, end))
                continue

            embeddings = [i[1] for i in chunks[start:end]]
            n_neighbors = int((len(embeddings) - 1) ** 0.8)
            reduced_embeddings = umap.UMAP(
                n_neighbors = max(2, n_neighbors),
                n_component = min(12, len(embeddings) - 2),
                metric = "cosine",
            ).fit_transform(embeddings)

# 不要在参数列表添加self类型的参数
# 改动参数名字很麻烦，尽量第一次就正确命名
            optimal_cluster = self._get_n_cluster(chunks[start, end],  random_state)

            if optimal_cluster == 1:
                lbls = [0 for _ in range(len(embeddings))]
            gm = GaussianMixture(optimal_cluster,random_state=random_state)
            gm.fit(reduced_embeddings)
            probs = gm.predict_proba(reduced_embeddings)
            lbls = [np.where(prob > self._threshold)[0] for prob in probs]
            lbls = [lbl[0] if isinstance(lbl, np.ndarray) else lbl for lbl in lbls]

            for i in range(optimal_cluster):
                idx = [j + start for j in range(len(lbls)) if lbls[j] == i] 
                self._summarize(chunks, idx)

            start = end
            end = len(chunks)
            layer.append((start, end))
            label.extend(lbls)
            # 因为是扁平化搜索，所以不需要树的结构信息
        return chunks






            