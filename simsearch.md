<header>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.css" integrity="sha384-yFRtMMDnQtDRO8rLpMIKrtPCD5jdktao2TV19YiZYWMDkUR5GQZR/NOVTdquEx1j" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.js" integrity="sha384-9Nhn55MVVN0/4OFx7EE5kpFBPsEMZxKTCnA+4fqDmg12eCTqGi6+BB2LjY8brQxJ" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
</header>

# Modèle de représentation de textes pour un prototype de recommandation: méthodologie et ingénierie

J'ai créé un prototype de recommandation de films sur base de transformers. Le titre de ma maquette [Canap' is all you need](https://huggingface.co/spaces/mnemlaghi/canap) est un clin d'oeil/hommage appuyé à l'article fondateur des mécanismes d'attention [Attention is all you need](https://arxiv.org/abs/1706.03762). J'évoque ici les différents modèles utilisés ainsi que quelques pistes d'exploitation.

## Principes méthodologiques

Les modèles de recommandation basés sur le contenu textuel se basent sur des modèles de représentation des textes. Concrètement,  un utilisateur rentre une requête courte comme une séquence de mots-clés. Cette requête est projetée (_encodée_) dans un espace de représentation au sein duquel nous pouvons effectuer des opérations algébrique: ainsi, la projection de cette requête est comparée avec les encodages de descriptifs de contenus. Cette méthode de comparaison donne un scoring final, qui va trier les contenus en fonction de leur similarité à la requête.

### Espaces de représentation

L'espace de représentation sert à calculer un score de similarité entre la requête et chacun des contenus. Le choix d'un tel espace de représentation est donc crucial en ce qui concerne la qualité de la recommandation subséquente. Formellement, si l'on a une suite de documents \\(( t_1, ...t_N)\\) issus d'un ensemble de documents que l'on souhaite comparer à une requête $q$ introduite par un utilisateur.

Le but du ranking est d'ordonner $N$ scores $s_i$, calculés via la démarche suivante:

 * les documents et la requête sont projetés dans un espace euclidien, i.e. une fonction $e_{\theta}$ qui projette les textes dans un espace hilbertien $\mathbf{R}^d$, où $d$ est la dimension de l'espace et $\theta$ un ensemble de paramètres caractérisant la projection $e$ :

 * Munis de cette projection $e$, nous pouvons calculer les scores suivants et d'un opérateur $<., .>$ :

$$ <.,.> : \mathbf{R}^d \times \mathbf{R}^d \rightarrow \mathbf{R} $$
$$ \forall i \in [[1,N]], s_i = <e_{\theta}(t_i), e_{\theta}(q)> $$

* L'opérateur $<.,.>$  peut être un opérateur comme le produit scalaire, la similarité cosine, etc. 

L'ensemble $\theta$ de paramètres peut être appris via de l'apprentissage supervisé

Nous proposons ici de comparer 3 types d'espace de représentation.

### Les espaces de représentation sparse

Une méthode sparse : le scoring est effectué à l'aide d'un espace de représentation _sparse_. Concrètement, nous proposons un ranking via les fréquences et les occurrences des mots contenus dans la requête et les descriptifs. L'espace de projection $d$ est équivalent à la cardinalité $V$ du vocabulaire (d'un ordre de grandeur se situant entre 10 000 et 1 000 000). Dans la maquette, nous avons choisi [TF IDF](https://fr.wikipedia.org/wiki/TF-IDF) avec prise en compte des bigrammes et filtrage des mots dont la fréquence est inférieure à 3.

### Les méthodes denses

Ici, l'espace de représentation est obtenu pour effectuer des opérations sur des aspects sémantiques (liés au sens des mots), plutôt que fréquentiels (liés aux occurrences strictes des mots). L'espace de représentation est ici appelé un embedding, car il représente une séquence dans un espace de dimension $d$ assez réduite pour pouvoir efficacement effectuer des opérations algébriques : $d << V$, où $V est le vocabulaire du corpus étudié.

#### Les embeddings _context free_

Au sein de ces méthodes denses, il s'agit en premier lieu de distinguer les embeddings _context free_ tels que [Glove](https://nlp.stanford.edu/projects/glove/) ou [Word2Vec](https://fr.wikipedia.org/wiki/Word2vec) - la représentation d'une séquence n'est alors qu'une opération sur les mots de la séquence.

#### Les transformers comme embeddings _context aware_

Nous distinguons également les embeddings _context aware_, tels que les transformers. Ceux-ci vont, grâce aux mécanismes d'attention, détecter par exemple la polysémie d'un mot.

Nous pouvons directement utiliser un transformer préentraîné, sans finetuning, en l'occurrence un modèle obtenu par [Semantic Bert](https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models). Le modèle pré-entraîné choisi apparaît comme étant un compromis entre qualité de l'output et vitesse d'encoding. En l'occurrence, j'ai choisi  [multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1) pour sa légèreté dans le prototypage.


#### Methode sparse ranking/dense reranking

Une méthode sparse ranking/dense reranking : la requête est plongée dans un espace de représentation _sparse_ via TF IDF, un premier filtrage est effectué sur un top 1000, puis un nouveau scoring est implémenté au sein d'un espace dense du transformer souhaité (reranking)

Les méthodes d'extraction dense  avec attention font l'objet d'un champ actif de recherche, comme en témoigne le challenge [MS Marco poussé par Microsoft](https://arxiv.org/pdf/2105.04021.pdf). Le sujet est d'obtenir un compromis entre qualité des résultats et vitesse de calcul. La qualité du calcul se matérialise par un score d'évaluation. Une métrique telle que le [Mean Reciprocal Rank](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) semble actuellement faire consensus

Similairement, si l'espace de représentation est important, la manière de calculer la proximité entre un document et une requête l'est également. Ce [post](https://medium.com/@kunal_gohrani/different-types-of-distance-metrics-used-in-machine-learning-e9928c5e26c7) en propose un tour d'horizon.

### Evaluation par apprentissage/finetuning

Si nous ne sommes pas satisfaits des espaces pré-entraînés, nous pouvons les affiner en ajustant tout ou partie de leur jeu de paramètres $\theta$ par un mécanisme d'apprentissage supervisé.

Pour ce faire, nous avons classiquemetn besoin d'un jeu d'exemples annotés ainsi que d'une fonction de coût. Celle-ci nous permettra de minimiser l'erreur liée à la construction de notre espace de projection. La littérature nous fournit entre autres deux possibilités, impliquant quelques ajustements quant à la donnée d'entraînement. Supposons que l'on dispose de M examples de requêtes $q_i$, pour $i \in [[1, M]]$ avec les résultats de documents, positifs et négatifs. 

#### Les _triplet loss_

Les triplets permettent de changer l'espace de représentation pour rapprocher les éléments similaires et éloigner les éléments dissemblables. La contrepartie est que chaque exemple doit être accompagné de deux exemples, l'un positif et l'autre négatif.
 
Une requête $q_i$ est alors qualifiée d'_anchor_ (ancre) , $t_{p}^i$ un texte correspondant à une réponse positive et $t_{n}^i$ une réponse négative. La fonction de coût - à minimiser en fonction des paramètres s'écrit alors :

$$ \mathcal{l}_{\theta}^{T}(q_i, t_{p}^i, t_{n}^i) = max(\left\lVert e_{\theta}(q_i) -  e_{\theta}(t_{p}^i) \right\lVert^2 - \left\lVert  e_{\theta}(q_i) - e_{\theta}(t_{n}^i)  \right\lVert^2, \epsilon)  $$

Le but de l'apprentissage est de trouver le jeu $\theta^*$ de paramètres satisfaisant : 

$$ \theta^* = \arg\min_{\theta}  \sum_{i=1}^{M}  \mathcal{l}_{\theta}^{T}(q_i, t_{p}^i, t_{n}^i) $$

$\left\lVert.\right\lVert$ est une norme euclidienne, et $\epsilon$ est qualifié de _marge_.

Pour chaque example, en fonction de la valeur retournée, nous distinguons trois types de triplets : 

* Easy triplets : ce sont des examples conduisant à $\mathcal{l}=0$. Ils ne permettent pas un apprentissage adéquat
* Hard triplets : ce sont des exemples "contradictoires", dans la mesure où la distance de la requête à l'exemple négatif est inférieur à la distance à l'exemple positif
* Semi hard triplets : la loss reste positive car le résultat est ambigü (dans la marge entre l'exemple positif et l'exemple négatif) 

#### Contrastive Loss

Ici, un exemple d'apprentissage se présente simplement sous la forme de trois entrées : la requête $q_i$, un exemple de document $t_i$, et un label $y_i$ siginifiant si l'exemple est négatif ($y_i=0$) ou positif ($y_i=1$).

Cette loss - appelons la $\mathcal{l}_{\theta}^{C}$  - s'écrit alors : 


$$ \mathcal{l}_{\theta}^{C}(q_i, t_i, y_i) = y_i\mathcal{l}_{\theta}^{+}(q_i, t_i)+ (1 - y_i)\mathcal{l}_{\theta}^{-}(q_i, t_i, \epsilon)   $$

Cette fonction prendra donc une forme différente selon si l'exemple est négatif ou positif. Nous allons nous aider de la fonction auxiliaire $E$, qui n'est autre qu'un produit scalaire normalisé. Il s'agit d'une définition de la similarité cosine : 

$$ E(x,y) =  \frac{ <x,y> }{\left\lVert x \right\lVert \left\lVert y \right\lVert}  $$

| Cas positif | Cas négatif |
|-------------|-------------|
| $$ \mathcal{l}_{\theta}^{+}(q_i, t_i) = \frac{1}{4}(1 - E(e_{\theta}(q_i), e_{\theta}(t_i))) $$ | $$ \mathcal{l}_{\theta}^{-}(q_i, t_i, \epsilon) = E(e_{\theta}(q_i), e_{\theta}(t_i))^{2}\mathbb{1}_{E(e_{\theta}(q_i), e_{\theta}(t_i))<\epsilon} $$ |

Ainsi, cette fonction de coût s'apprend de la même manière que précédemment : 

$$ \theta^* = \arg\min_{\theta}  \sum_{i=1}^{M}  \mathcal{l}_{\theta}^{C}(q_i, t_i) $$

## Ingénierie : pistes d'industrialisation (en cours 🚧 🚧 🚧)

 La profondeur de la base de données utilisée pour la maquette pour ce prototype est assez faible pour encore implémenter un moteur industrialisable. Néanmoins, quelques pistes peuvent être suggérées en guise d'industrialisation.

#### Pistes d'indexation scalable.

L'indexation consiste à relier une séquence éligible à la recherche en un vecteur requêtable par suite.

Au sein des méthodes d'extraction sparse, une foultitude d'autres possibilités que le TF IDF existe. En l'occurrence, Elastic utilise par défaut [BM25](https://www.elastic.co/fr/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables)
Evoquons un instant les méthodes d'indexation et/ou de requêtage. En effet, si, dans notre cas, les index (ce qui va effectivement relier un document à sa représentation) créés sont de taille relativement faible, dans la pratique, une opération comparant une requête utilisateur à des millions, voire des millards d'entités se trouve être un challenge technique d'envergure. Le logiciel [ElasticSearch](https://www.elastic.co/fr/elasticsearch/) est une réussite en ce qui concerne la scalabilité. 


Une méthode d'indexation efficace peut se faire à l'aide de [FAISS](https://github.com/facebookresearch/faiss).

### Considérations d'infrastructure : de l'utilité du cloud

Pour des raisons d'élasticité, autant lors de la formation du modèle que lors de la mise à disposition des requêtes, nous pouvons utiliser à profit les utilitaires du cloud. Par exemple, la donnée peut être hebergée dans un bucket AWS S3 : la visite d'un utilisateur et sa requête enclenchent AWS Lambda, qui lui-même appelle un [endpoint SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html), contenant le modèle en production. Suivant le modèle utilisé et/ou la taille de la base de données, cet endpoint peut s'appuyer sur des instances plus ou moins puissantes en termes de capacité de calcul


## Quelques ressources

* Bien sûr, il y a le nécessaire [blog de Jay Alammar](https://jalammar.github.io/illustrated-bert/) pour comprendre BERT et les Transformers.
* Elastic.co fait un excellent travail didactique de présentation du _revelance scoring_ ([ici](https://www.elastic.co/fr/blog/practical-bm25-part-1-how-shards-affect-relevance-scoring-in-elasticsearch) et [là](https://www.elastic.co/fr/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables)
* Cet [article du MIT](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00369/100684/Sparse-Dense-and-Attentional-Representations-for) formalise non seulement de manière très rigoureuse le requêtage de textes, mais conduit également une étude exhaustive des possibilités.
* Sur les fonctions de coût permettant d'appréhender la similarité : [triplet loss](https://arxiv.org/pdf/1503.03832.pdf) et [contrastive loss](https://aclanthology.org/W16-1617.pdf)
