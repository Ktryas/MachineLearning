# 📚 Fiche Complète pour l'Examen de Machine Learning

**Objectif** : Une fiche ultra-complète basée uniquement sur tes TP/notebooks, avec :
- ✅ Toutes les fonctions utilisées dans tes notebooks
- ✅ Formules mathématiques + code NumPy
- ✅ Explications détaillées des graphiques
- ✅ Méthodes de chargement/découpage des données
- ✅ Structurée par concept pour répondre à n'importe quelle question d'examen

**Bibliothèques autorisées** (d'après tes TP) : `numpy`, `matplotlib`, `scipy`, `pandas`

**À faire sur machine** : Ce notebook peut être ouvert dans VS Code avec le TensorFlow du professeur.

---

## 📌 Sommaire

1. [📥 Chargement et Prétraitement des Données](#1-📥-chargement-et-prétraitement-des-données)
2. [🧮 Régression Linéaire](#2-🧮-régression-linéaire)
3. [🎯 Régression Logistique](#3-🎯-régression-logistique)
4. [🤖 Réseaux de Neurones](#4-🤖-réseaux-de-neurones)
5. [⚙️ Optimisation](#5-⚙️-optimisation)
6. [📊 Évaluation et Diagnostics](#6-📊-évaluation-et-diagnostics)
7. [🔍 Études de Cas (Exemples Concrets)](#7-🔍-études-de-cas)
8. [❓ Questions Fréquentes en Examen](#8-❓-questions-fréquentes-en-examen)

---

## 💡 Conseils pour l'Examen

- **Vérifie toujours les dimensions** de tes matrices avec `X.shape`, `y.shape`, `theta.shape`
- **Normalise tes données** si nécessaire (sauf si déjà normalisées)
- **Initialise bien tes poids** (évite les zéros pour la régression logistique/NN)
- **Trace les courbes d'apprentissage** pour détecter overfitting/underfitting
- **Vérifie la convergence** de ta fonction de coût

---

# 1. 📥 Chargement et Prétraitement des Données

> **Pourquoi c'est important** : 80% des erreurs en ML viennent d'un mauvais prétraitement des données.

## 1.1 Chargement des Données

### 📄 Fichiers `.txt` (ex: `ex1data1.txt`, `ex2data1.txt`)
Format : données séparées par des virgules, sans en-tête.

```python
import numpy as np

# Charger un fichier .txt avec numpy
data = np.loadtxt('ex1data1.txt', delimiter=',')

# Séparer X (features) et y (labels)
X = data[:, :-1]  # Toutes les colonnes sauf la dernière
y = data[:, -1]   # Dernière colonne

# Afficher les dimensions
m = len(y)  # Nombre d'exemples
n = X.shape[1]  # Nombre de features
print(f"X shape: {X.shape} (m={m}, n={n})")
print(f"y shape: {y.shape}")
```

**Explications** :
- `np.loadtxt()` charge un fichier texte dans un tableau NumPy
- `delimiter=','` spécifie que les valeurs sont séparées par des virgules
- `X = data[:, :-1]` prend toutes les lignes et toutes les colonnes sauf la dernière
- `y = data[:, -1]` prend uniquement la dernière colonne (labels)
- **Question d'examen typique** : "Quelle est la dimension de X et de y ?" → Réponse : `X.shape = (m, n)`, `y.shape = (m,)`

---

### 📄 Fichiers `.mat` (ex: `ex3data1.mat`)
Format : données MATLAB sauvegardées avec scipy.

```python
from scipy.io import loadmat

# Charger un fichier .mat
data = loadmat('ex3data1.mat')

# Extraire X et y (les clés dépendent du fichier)
X = data['X']  # ou data['X_train'] selon le fichier
y = data['y']  # ou data['y_train']

# Vérifier les dimensions
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Si y est en format vectorisé (1D array), le convertir en colonne
if y.ndim == 1:
    y = y.reshape(-1, 1)
```

**Explications** :
- `loadmat()` charge un fichier MATLAB et retourne un dictionnaire
- Les clés du dictionnaire dépendent de comment les données ont été sauvegardées
- **Astuce** : Utilise `data.keys()` pour voir les variables disponibles
- **Attention** : Dans certains TP, y est stocké comme un vecteur ligne (1D), il faut le reshaper en colonne

---

### 📄 Fichiers `.csv` (ex: `student_habits_performance.csv`)
Format : données avec en-tête, séparées par des virgules.

```python
import pandas as pd

# Charger un fichier CSV avec pandas
df = pd.read_csv('student_habits_performance.csv')

# Convertir en numpy arrays
X = df.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière
y = df.iloc[:, -1].values   # Dernière colonne

# Vérifier les dimensions
print(f"DataFrame shape: {df.shape}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
```

**Explications** :
- `pd.read_csv()` charge un CSV dans un DataFrame pandas
- `.iloc[:, :-1]` sélectionne toutes les lignes et colonnes sauf la dernière
- `.values` convertit le DataFrame en tableau NumPy
- **Avantage** : Pandas gère automatiquement les en-têtes

---

## 1.2 Normalisation des Données

> **Pourquoi normaliser ?** : La descente de gradient converge plus vite si les features sont à la même échelle.

```python
def feature_normalize(X):
    """
    Normalise les features de X (Mean Normalization + Feature Scaling)
    
    Formule : X_norm = (X - μ) / σ
    où μ = moyenne, σ = écart-type
    """
    mu = np.mean(X, axis=0)  # Moyenne de chaque feature
    sigma = np.std(X, axis=0)  # Écart-type de chaque feature
    
    # Normalisation
    X_norm = (X - mu) / sigma
    
    return X_norm, mu, sigma

# Exemple d'utilisation
X_norm, mu, sigma = feature_normalize(X)
print(f"Moyennes: {mu}")
print(f"Écarts-types: {sigma}")
```

**Formule Mathématique** :

$$
\newcommand{\mu}{\mu}
\newcommand{\sigma}{\sigma}
\newcommand{\x}{\mathbf{x}}
\newcommand{\X}{\mathbf{X}}
$$

Pour chaque feature $j$ :

$$
 x^{(i)}_{norm,j} = \frac{x^{(i)}_{j} - \mu_j}{\sigma_j}
$$

Où :
- $x^{(i)}_{j}$ = valeur de la feature $j$ pour l'exemple $i$
- $\mu_j = \frac{1}{m}\sum_{i=1}^{m} x^{(i)}_{j}$ = moyenne de la feature $j$
- $\sigma_j = \sqrt{\frac{1}{m}\sum_{i=1}^{m} (x^{(i)}_{j} - \mu_j)^2}$ = écart-type de la feature $j$

**Code NumPy équivalent** :
```python
mu = np.mean(X, axis=0)      # Moyenne par colonne
sigma = np.std(X, axis=0)    # Écart-type par colonne
X_norm = (X - mu) / sigma    # Normalisation
```

**Cas particuliers** :
- Si $\sigma_j = 0$ (feature constante) → **ne pas normaliser** (division par zéro)
- Pour la régression logistique, **toujours normaliser** sauf le terme constant (bias)

---

## 1.3 Découpage Train/Validation/Test

> **Pourquoi découper ?** : Évaluer les performances du modèle sur des données non vues pendant l'entraînement.

```python
# Méthode 1 : Découpage manuel (sans scikit-learn)
np.random.seed(42)  # Pour la reproductibilité

# Mélanger les indices
indices = np.random.permutation(m)

# Proportions typiques
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Calculer les tailles
train_size = int(m * train_ratio)
val_size = int(m * val_ratio)

# Découper
train_idx = indices[:train_size]
val_idx = indices[train_size:train_size+val_size]
test_idx = indices[train_size+val_size:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
```

**Explications** :
- `np.random.permutation(m)` mélange les indices de 0 à m-1
- **70% train, 15% validation, 15% test** est un split classique
- **Valider la taille** : `train_size + val_size + test_size == m`
- **Question d'examen** : "Pourquoi faut-il mélanger les données avant de découper ?" → Pour éviter un biais si les données sont ordonnées

---

## 1.4 Ajout du Terme Constant (Bias)

> **Pourquoi ?** : Le terme constant (θ₀) permet au modèle de s'ajuster même si toutes les features sont à 0.

```python
# Ajouter une colonne de 1 à X pour le terme constant
X_b = np.c_[np.ones((m, 1)), X]  # ou np.hstack([np.ones((m,1)), X])

# Vérifier la nouvelle dimension
print(f"X avec bias shape: {X_b.shape}")  # Devrait être (m, n+1)
```

**Explications** :
- `np.c_[]` concatène des tableaux le long du **2ème axe** (colonnes)
- `np.ones((m, 1))` crée une colonne de 1 de taille m
- **Dimension** : Si `X.shape = (m, n)`, alors `X_b.shape = (m, n+1)`
- **Formule** : Le modèle devient $h_\theta(x) = \theta_0 + \theta_1 x_1 + ... + \theta_n x_n$

---

# 2. 🧮 Régression Linéaire

> **Objectif** : Trouver la meilleure droite (ou hyperplan) qui minimise l'erreur entre les prédictions et les vraies valeurs.

## 2.1 Hypothèse et Modèle

### Hypothèse

$$
h_\theta(\x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n = \theta^T \x
$$

Où :
- $h_\theta(\x)$ = prédiction
- $\theta = [\theta_0, \theta_1, ..., \theta_n]^T$ = vecteur des paramètres
- $\x = [x_0, x_1, ..., x_n]^T$ avec $x_0 = 1$ (terme constant)

```python
def h(theta, X):
    """
    Calcule l'hypothèse : h_θ(X) = X @ θ
    
    Args:
        theta: vecteur des paramètres (n+1,)
        X: matrice des features avec bias (m, n+1)
    
    Returns:
        predictions: vecteur des prédictions (m,)
    """
    return X @ theta  # ou np.dot(X, theta)
```

**Formule en NumPy** :

$$
\hat{y} = X \cdot \theta
$$

Où :
- $X$ est de dimension $(m, n+1)$ (avec la colonne de 1 pour $\theta_0$)
- $\theta$ est de dimension $(n+1,)$
- Le résultat $\hat{y}$ est de dimension $(m,)$
- **Produit matriciel** : $X \cdot \theta = \sum_{j=0}^{n} X_{:,j} \cdot \theta_j$

---

## 2.2 Fonction de Coût (MSE - Mean Squared Error)

### Formule

```python
def compute_cost(X, y, theta):
    """
    Calcule la fonction de coût J(θ) pour la régression linéaire
    
    Formule : J(θ) = (1/(2m)) * Σ(h_θ(x^(i)) - y^(i))^2
    """
    m = len(y)
    predictions = X @ theta
    errors = predictions - y
    J = (1 / (2 * m)) * np.sum(errors**2)
    return J
```

**Formule Mathématique** :

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(\x^{(i)}) - y^{(i)})^2
$$

**Code NumPy vectorisé** :
```python
errors = X @ theta - y          # (m,) - (m,) = (m,)
squared_errors = errors ** 2     # Élève chaque élément au carré
J = np.sum(squared_errors) / (2 * m)  # Somme + division
```

**Remarques** :
- Le facteur 1/2 est pour simplifier la dérivée
- **Dimension de `errors`** : doit être (m,) pour que la somme fonctionne
- **Question d'examen** : "Pourquoi divise-t-on par 2m ?" → Pour que la dérivée soit propre : ∂J/∂θ = (1/m) * X^T @ (Xθ - y)

---

## 2.3 Descente de Gradient

### Formule de Mise à Jour

```python
def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Descente de gradient pour la régression linéaire
    
    Args:
        X: matrice des features avec bias (m, n+1)
        y: vecteur des labels (m,)
        theta: vecteur initial des paramètres (n+1,)
        alpha: learning rate
        num_iters: nombre d'itérations
    
    Returns:
        theta: paramètres optimisés
        J_history: historique de la fonction de coût
    
    Formule : θ := θ - (α/m) * X^T @ (Xθ - y)
    """
    m = len(y)
    J_history = []
    
    for _ in range(num_iters):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1 / m) * (X.T @ errors)  # Dérivée de J(θ)
        theta = theta - alpha * gradient
        J_history.append(compute_cost(X, y, theta))
    
    return theta, J_history
```

**Formules Mathématiques** :

Mise à jour simultanée de tous les θ :

$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta) \quad \text{pour tout } j = 0,1,...,n
$$

Dérivée partielle :

$$
\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(\x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}
$$

**Formule vectorisée** (pour tous les θ en une fois) :

$$
\nabla J(\theta) = \frac{1}{m} X^T (X \theta - \mathbf{y})
$$

**Code NumPy** :
```python
gradient = (1/m) * X.T @ (X @ theta - y)  # (n+1,) = (n+1, m) @ (m,)
theta = theta - alpha * gradient
```

**Explications** :
- `X.T @ errors` = $X^T \cdot (X\theta - y)$ (produit matriciel)
- **Dimension de `gradient`** : doit être `(n+1,)` pour correspondre à `theta`
- **Learning rate (α)** : Trop grand → divergence, trop petit → lent
- **Nombre d'itérations** : Jusqu'à convergence (quand J(θ) ne change plus beaucoup)

```python
# Exemple d'utilisation complet pour TP1

# 1. Charger les données
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, 0]
y = data[:, 1]
m = len(y)

# 2. Ajouter le terme constant
X_b = np.c_[np.ones((m, 1)), X]

# 3. Initialiser theta à zéro
theta = np.zeros(2)

# 4. Lancer la descente de gradient
alpha = 0.01
num_iters = 1500
theta_opt, J_history = gradient_descent(X_b, y, theta, alpha, num_iters)

# 5. Afficher le coût final
print(f"Coût final: {J_history[-1]:.4f}")
print(f"Paramètres optimaux: {theta_opt}")

# 6. Tracer la courbe de coût
plt.plot(J_history)
plt.xlabel('Itérations')
plt.ylabel('J(θ)')
plt.title('Convergence de la descente de gradient')
plt.show()
```

### 📈 Interprétation du Graphique de Coût

**Ce que tu dois voir** :
- Une courbe qui **décroît** à chaque itération
- Une **convergence** vers une valeur minimale (plateau)

**Problèmes courants** :
- **Courbe qui oscille** → Learning rate (α) **trop grand** → Réduire α
- **Courbe qui décroît trop lentement** → Learning rate (α) **trop petit** → Augmenter α
- **Courbe qui ne décroît pas** → Problème de normalisation ou de dimensions

**Question d'examen** : "Comment choisir le bon learning rate ?" →
Essayer plusieurs valeurs (0.001, 0.01, 0.1, 1) et tracer J(θ) pour voir laquelle converge le mieux.

---

# 3. 🎯 Régression Logistique

> **Objectif** : Classifier des données binaires (0 ou 1) en utilisant la fonction sigmoïde.

## 3.1 Fonction Sigmoïde

### Formule

```python
def sigmoid(z):
    """
    Fonction sigmoïde : g(z) = 1 / (1 + e^(-z))
    
    Args:
        z: scalaire, vecteur ou matrice
    
    Returns:
        g(z): entre 0 et 1
    """
    return 1 / (1 + np.exp(-z))
```

**Formule Mathématique** :

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

**Propriétés** :
- $g(z) \in (0, 1)$ pour tout $z \in \mathbb{R}$
- $g(0) = 0.5$
- $g(z) \to 1$ quand $z \to +\infty$
- $g(z) \to 0$ quand $z \to -\infty$
- **Symétrie** : $g(-z) = 1 - g(z)$

**Graphique** :

```python
z = np.linspace(-10, 10, 100)
plt.plot(z, sigmoid(z))
plt.axhline(0.5, color='r', linestyle='--')
plt.xlabel('z')
plt.ylabel('g(z)')
plt.title('Fonction Sigmoïde')
plt.show()
```

**Interprétation** :
- La sigmoïde **compresse** n'importe quelle valeur réelle entre 0 et 1
- Peut être interprétée comme une **probabilité**
- **Seuil à 0.5** : Si $g(z) \geq 0.5$, prédire la classe 1, sinon 0

---

## 3.2 Hypothèse de la Régression Logistique

```python
def h_logistic(theta, X):
    """
    Hypothèse pour la régression logistique
    
    h_θ(x) = sigmoid(X @ θ)
    """
    return sigmoid(X @ theta)
```

**Formule** :

$$
h_\theta(\x) = g(\theta^T \x) = \frac{1}{1 + e^{-\theta^T \x}}
$$

**Interprétation** :
- $h_\theta(\x)$ = probabilité que $y = 1$ donné $x$
- $P(y=1|x;\theta) = h_\theta(\x)$
- $P(y=0|x;\theta) = 1 - h_\theta(\x)$

---

## 3.3 Fonction de Coût pour la Régression Logistique

### Formule

```python
def compute_cost_logistic(X, y, theta):
    """
    Fonction de coût pour la régression logistique
    
    Formule : J(θ) = (-1/m) * Σ[y*log(h) + (1-y)*log(1-h)]
    """
    m = len(y)
    h = sigmoid(X @ theta)
    
    # Éviter log(0) en ajoutant un epsilon
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1 - epsilon)
    
    J = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return J
```

**Formule Mathématique** :

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(\x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(\x^{(i)})) \right]
$$

**Explications** :
- **Log Vraisemblance** : Cette fonction de coût vient de la maximisation de la vraisemblance
- **Problème numérique** : Si $h_\theta(x) = 0$ ou $1$, $\log(0)$ est indéfini → **clipper les valeurs**
- **Dimension** : `y * np.log(h)` = (m,) * (m,) = (m,)

**Code NumPy** :
```python
h = sigmoid(X @ theta)  # (m,)
term1 = y * np.log(h)     # y=1: contribue, y=0: 0
term2 = (1-y) * np.log(1-h)  # y=0: contribue, y=1: 0
J = -np.sum(term1 + term2) / m
```

---

## 3.4 Descente de Gradient pour la Régression Logistique

```python
def gradient_descent_logistic(X, y, theta, alpha, num_iters):
    """
    Descente de gradient pour la régression logistique
    
    Formule : θ := θ - (α/m) * X^T @ (h_θ(X) - y)
    """
    m = len(y)
    J_history = []
    
    for _ in range(num_iters):
        h = sigmoid(X @ theta)
        errors = h - y  # Différence entre prédiction et vraie valeur
        gradient = (1 / m) * (X.T @ errors)
        theta = theta - alpha * gradient
        J_history.append(compute_cost_logistic(X, y, theta))
    
    return theta, J_history
```

**Formule de la Dérivée** :

$$
\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(\x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}
$$

**Formule Vectorisée** :

$$
\nabla J(\theta) = \frac{1}{m} X^T (h_\theta(X) - \mathbf{y})
$$

**Remarque** : La formule est **identique** à celle de la régression linéaire, mais avec $h_\theta(x) = g(X\theta)$ au lieu de $h_\theta(x) = X\theta$

```python
# Exemple pour TP2 (ex2data1.txt)

# 1. Charger les données
data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]
m = len(y)

# 2. Normaliser les données
X_norm, mu, sigma = feature_normalize(X)

# 3. Ajouter le terme constant
X_b = np.c_[np.ones((m, 1)), X_norm]

# 4. Initialiser theta à zéro
theta = np.zeros(3)

# 5. Lancer la descente de gradient
alpha = 0.1
num_iters = 400
theta_opt, J_history = gradient_descent_logistic(X_b, y, theta, alpha, num_iters)

# 6. Afficher les résultats
print(f"Coût final: {J_history[-1]:.4f}")
print(f"Paramètres optimaux: {theta_opt}")

# 7. Tracer la frontière de décision
plt.scatter(X_norm[y==1, 0], X_norm[y==1, 1], color='red', marker='*', label='Admis')
plt.scatter(X_norm[y==0, 0], X_norm[y==0, 1], color='yellow', edgecolor='black', marker='o', label='Non admis')

# Créer une grille pour la frontière
x1_min, x1_max = X_norm[:, 0].min(), X_norm[:, 0].max()
x2_min, x2_max = X_norm[:, 1].min(), X_norm[:, 1].max()
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))

# Prédire pour chaque point de la grille
X_grid = np.c_[np.ones((100*100, 1)), xx1.ravel(), xx2.ravel()]
preds = (sigmoid(X_grid @ theta_opt) >= 0.5).astype(int)
preds = preds.reshape(xx1.shape)

plt.contourf(xx1, xx2, preds, alpha=0.2, colors=['yellow', 'red'])
plt.xlabel('Score Examen 1 (normalisé)')
plt.ylabel('Score Examen 2 (normalisé)')
plt.legend()
plt.title('Frontière de décision - TP2')
plt.show()
```

### 📈 Interprétation de la Frontière de Décision

**Ce que tu dois voir** :
- Une **ligne droite** (ou courbe) qui sépare les classes 0 et 1
- Les points **rouges** (classe 1) d'un côté, les **jaunes** (classe 0) de l'autre

**Problèmes courants** :
- **Frontière mal placée** → Le modèle n'a pas convergé → Augmenter `num_iters` ou ajuster `alpha`
- **Frontière trop complexe** → Overfitting → Ajouter de la régularisation
- **Frontière trop simple** → Underfitting → Ajouter des features (polynômes)

---

# 4. 🤖 Réseaux de Neurones

> **Objectif** : Implémenter un réseau de neurones pour des tâches de classification plus complexes.

## 4.1 Architecture du Réseau

### Schéma

Couche d'entrée → Couche cachée → Couche de sortie

Exemple pour TP3 (ex3data1.mat) :
- **Couche d'entrée** : 400 neurones (pixels 20x20)
- **Couche cachée** : 25 neurones
- **Couche de sortie** : 10 neurones (chiffres 0-9)

```python
# Définition de l'architecture
input_layer_size = 400   # 20x20 pixels
hidden_layer_size = 25  # Nombre de neurones dans la couche cachée
output_layer_size = 10  # 10 classes (chiffres 0-9)

print(f"Architecture: {input_layer_size} -> {hidden_layer_size} -> {output_layer_size}")
```

**Notation** :

- $L$ = nombre de couches (ex: L=3 pour entrée+cachée+sortie)
- $s_l$ = nombre de neurones dans la couche $l$ (s₁=input, s₂=hidden, s₃=output)
- $\Theta^{(l)}$ = matrice des poids entre la couche $l$ et $l+1$ (dimension : $s_{l+1} \times (s_l + 1)$)
  - **+1** pour le terme de bias

**Exemple pour TP3** :
- $\Theta^{(1)}$ : poids entre entrée et cachée → dimension $(25, 401)$
- $\Theta^{(2)}$ : poids entre cachée et sortie → dimension $(10, 26)$

---

## 4.2 Initialisation des Poids

### Pourquoi pas des zéros ?
Si tous les poids sont initialisés à 0, tous les neurones d'une couche auront la même valeur → **symétrie brisée** → le réseau n'apprend pas.

```python
def initialize_weights(input_size, hidden_size, output_size):
    """
    Initialise les poids aléatoirement avec une petite variance
    
    Méthode : Initialisation aléatoire entre -ε et ε
    où ε = sqrt(6 / (fan_in + fan_out)) (Xavier/Glorot)
    """
    # Initialisation pour la couche entrée → cachée
    epsilon_init = 0.12  # Valeur typique pour les petits réseaux
    Theta1 = np.random.randn(hidden_size, input_size + 1) * epsilon_init
    
    # Initialisation pour la couche cachée → sortie
    Theta2 = np.random.randn(output_size, hidden_size + 1) * epsilon_init
    
    return Theta1, Theta2

# Exemple
Theta1, Theta2 = initialize_weights(input_layer_size, hidden_layer_size, output_layer_size)
print(f"Theta1 shape: {Theta1.shape}")  # (25, 401)
print(f"Theta2 shape: {Theta2.shape}")  # (10, 26)
```

**Formules d'Initialisation** :

1. **Initialisation aléatoire simple** :
   $$
   \Theta^{(l)} \sim \text{Uniform}(-\epsilon, \epsilon)
   $$
   avec $\epsilon = 0.12$ (valeur typique)

2. **Initialisation Xavier/Glorot** (recommandée) :
   $$
   \epsilon = \sqrt{\frac{6}{s_{l} + s_{l+1}}}
   $$
   où $s_l$ = nombre de neurones dans la couche $l$

**Pourquoi pas des zéros ?** :
- Si $\Theta^{(1)} = 0$, alors $a^{(2)} = g(0) = 0.5$ pour tous les neurones
- Tous les neurones de la couche cachée auront la même valeur → **pas d'apprentissage**
- L'initialisation aléatoire **brise la symétrie**

---

## 4.3 Forward Propagation

### Étapes
1. Calculer $z^{(2)} = \Theta^{(1)} \cdot a^{(1)}$
2. Calculer $a^{(2)} = g(z^{(2)})$ (fonction d'activation)
3. Ajouter le terme de bias à $a^{(2)}$
4. Calculer $z^{(3)} = \Theta^{(2)} \cdot a^{(2)}$
5. Calculer $a^{(3)} = g(z^{(3)})$ (sortie)

```python
def forward_propagation(X, Theta1, Theta2):
    """
    Effectue la forward propagation
    
    Args:
        X: données d'entrée (m, n)
        Theta1: poids couche 1 (hidden_size, input_size + 1)
        Theta2: poids couche 2 (output_size, hidden_size + 1)
    
    Returns:
        a3: prédictions finales (m, output_size)
        z2, a2, z3: valeurs intermédiaires pour le backprop
    """
    m = X.shape[0]
    
    # Ajouter le terme de bias à X (a1)
    a1 = np.c_[np.ones((m, 1)), X]  # (m, input_size + 1)
    
    # Calculer z2
    z2 = a1 @ Theta1.T  # (m, hidden_size)
    
    # Calculer a2 = g(z2)
    a2 = sigmoid(z2)
    
    # Ajouter le terme de bias à a2
    a2 = np.c_[np.ones((m, 1)), a2]  # (m, hidden_size + 1)
    
    # Calculer z3
    z3 = a2 @ Theta2.T  # (m, output_size)
    
    # Calculer a3 = g(z3) (pour la classification binaire)
    # Pour la classification multi-classe, utiliser softmax à la place
    a3 = sigmoid(z3)
    
    return a3, z2, a2, z3
```

**Formules Mathématiques** :

Pour chaque exemple $i$ :

1. **Couche cachée** :
   $$
   z^{(2)}_{i} = \Theta^{(1)} \cdot a^{(1)}_{i} \quad \text{où } a^{(1)}_{i} = \begin{bmatrix} 1 \\ x_{i} \end{bmatrix}
   $$
   $$
   a^{(2)}_{i} = g(z^{(2)}_{i})
   $$

2. **Couche de sortie** :
   $$
   z^{(3)}_{i} = \Theta^{(2)} \cdot a^{(2)}_{i} \quad \text{où } a^{(2)}_{i} = \begin{bmatrix} 1 \\ g(z^{(2)}_{i}) \end{bmatrix}
   $$
   $$
   a^{(3)}_{i} = g(z^{(3)}_{i})
   $$

**Remarques** :
- $a^{(1)}$ = couche d'entrée + bias (x₀=1)
- $a^{(2)}$ = couche cachée + bias
- $a^{(3)}$ = couche de sortie (probabilités)

---

## 4.4 Fonction de Coût pour les Réseaux de Neurones

```python
def compute_cost_nn(X, y, Theta1, Theta2, output_size):
    """
    Fonction de coût pour un réseau de neurones avec régularisation
    
    Formule : J(θ) = (-1/m) * Σ[y*log(h) + (1-y)*log(1-h)] + (λ/(2m)) * ΣΘ^2
    """
    m = X.shape[0]
    lambda_reg = 0.1  # Paramètre de régularisation
    
    # Forward propagation
    a3, z2, a2, z3 = forward_propagation(X, Theta1, Theta2)
    
    # Calcul du coût (sans régularisation)
    epsilon = 1e-15
    a3 = np.clip(a3, epsilon, 1 - epsilon)
    
    # Pour la classification binaire
    J = (-1 / m) * np.sum(y * np.log(a3) + (1 - y) * np.log(1 - a3))
    
    # Ajouter la régularisation
    reg_term = (lambda_reg / (2 * m)) * (np.sum(Theta1[:, 1:]**2) + np.sum(Theta2[:, 1:]**2))
    J += reg_term
    
    return J
```

**Formule avec Régularisation** :

$$
J(\Theta) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} \left[ y_k^{(i)} \log((h_\Theta(x^{(i)}))_k) \right] + \frac{\lambda}{2m} \sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} (\Theta_{ji}^{(l)})^2
$$

**Explications** :
- **Premier terme** : Coût de classification (cross-entropy pour multi-classe)
- **Deuxième terme** : Régularisation L2 (évite l'overfitting)
- **λ (lambda)** : Paramètre de régularisation (0 = pas de régularisation)
- **Ne pas régulariser le bias** : Les colonnes 0 de Θ (bias) ne sont pas incluses dans la régularisation

**Code NumPy** :
```python
# Régularisation (sans le bias)
reg_term = (lambda_reg / (2 * m)) * (
    np.sum(Theta1[:, 1:]**2) +  # Tous les poids sauf la 1ère colonne (bias)
    np.sum(Theta2[:, 1:]**2)
)
```

---

## 4.5 Backpropagation

### Principe
Calculer les gradients en **remontant** les erreurs depuis la couche de sortie vers la couche d'entrée.

```python
def backpropagation(X, y, Theta1, Theta2, lambda_reg=0.1):
    """
    Implémente la backpropagation
    
    Returns:
        D1: gradient pour Theta1
        D2: gradient pour Theta2
    """
    m = X.shape[0]
    output_size = Theta2.shape[0]
    
    # Forward propagation
    a1 = np.c_[np.ones((m, 1)), X]
    z2 = a1 @ Theta1.T
    a2 = sigmoid(z2)
    a2 = np.c_[np.ones((m, 1)), a2]
    z3 = a2 @ Theta2.T
    a3 = sigmoid(z3)
    
    # Calcul de l'erreur
    delta3 = a3 - y  # (m, output_size)
    
    # Backpropagation pour Theta2
    D2 = (1 / m) * (delta3.T @ a2)  # (output_size, hidden_size + 1)
    D2[:, 1:] += (lambda_reg / m) * Theta2[:, 1:]  # Régularisation (sans bias)
    
    # Backpropagation pour Theta1
    delta2 = (delta3 @ Theta2[:, 1:]) * sigmoid_gradient(z2)  # (m, hidden_size)
    D1 = (1 / m) * (delta2.T @ a1)  # (hidden_size, input_size + 1)
    D1[:, 1:] += (lambda_reg / m) * Theta1[:, 1:]  # Régularisation (sans bias)
    
    return D1, D2

def sigmoid_gradient(z):
    """Dérivée de la sigmoïde : g'(z) = g(z) * (1 - g(z))"""
    g = sigmoid(z)
    return g * (1 - g)
```

**Formules de Backpropagation** :

1. **Erreur de sortie** :
   $$
   \delta^{(L)} = a^{(L)} - y
   $$

2. **Erreur couche cachée** :
   $$
   \delta^{(l)} = (\Theta^{(l)})^T \cdot \delta^{(l+1)} \odot g'(z^{(l)})
   $$
   où $\odot$ = produit élément par élément (Hadamard)

3. **Gradient pour Θ** :
   $$
   D^{(l)} = \frac{1}{m} \delta^{(l+1)} \cdot (a^{(l)})^T
   $$

**Avec régularisation** :
$$
D^{(l)}_{ji} = D^{(l)}_{ji} + \frac{\lambda}{m} \Theta^{(l)}_{ji} \quad \text{pour } j \geq 1
$$
(Ne pas régulariser les termes de bias, $j=0$)

---

## 4.6 Entraînement du Réseau

```python
def train_nn(X, y, input_size, hidden_size, output_size, alpha=0.1, num_iters=100, lambda_reg=0.1):
    """
    Entraîne un réseau de neurones avec une couche cachée
    """
    # Initialiser les poids
    Theta1, Theta2 = initialize_weights(input_size, hidden_size, output_size)
    
    J_history = []
    
    for _ in range(num_iters):
        # Calculer le coût
        J = compute_cost_nn(X, y, Theta1, Theta2, output_size)
        J_history.append(J)
        
        # Backpropagation
        D1, D2 = backpropagation(X, y, Theta1, Theta2, lambda_reg)
        
        # Mise à jour des poids
        Theta1 -= alpha * D1
        Theta2 -= alpha * D2
    
    return Theta1, Theta2, J_history
```

**Exemple pour TP3 (ex3data1.mat)** :

```python
# Charger les données
data = loadmat('ex3data1.mat')
X = data['X']  # (5000, 400)
y = data['y']  # (5000,)

# Convertir y en one-hot encoding
y_onehot = np.eye(10)[y.astype(int).flatten()]  # (5000, 10)

# Entraîner le réseau
Theta1, Theta2, J_history = train_nn(
    X, y_onehot, input_size=400, hidden_size=25, output_size=10,
    alpha=0.1, num_iters=100, lambda_reg=0.1
)

# Tracer la courbe de coût
plt.plot(J_history)
plt.xlabel('Itérations')
plt.ylabel('J(Θ)')
plt.title('Convergence - TP3')
plt.show()

# Prédire sur un exemple
def predict_nn(X, Theta1, Theta2):
    a3, _, _, _ = forward_propagation(X, Theta1, Theta2)
    return np.argmax(a3, axis=1)

# Prédire le premier exemple
sample_idx = 0
pred = predict_nn(X[sample_idx:sample_idx+1], Theta1, Theta2)
print(f"Vrai label: {y[sample_idx]}, Prédiction: {pred[0]}")
```

---

# 5. ⚙️ Optimisation

> **Objectif** : Trouver les meilleurs paramètres θ qui minimisent la fonction de coût J(θ).

## 5.1 Choix du Learning Rate (α)

### Comment choisir α ?

| Valeur de α | Comportement | Solution |
|--------------|--------------|----------|
| Trop grand (ex: 10) | Divergence (J(θ) → ∞) | Réduire α |
| Trop petit (ex: 0.00001) | Convergence très lente | Augmenter α |
| Bon (ex: 0.01, 0.1) | Convergence rapide | Garder |

**Méthode recommandée** :
1. Essayer α = 0.01, 0.03, 0.1, 0.3, 1.0
2. Tracer J(θ) pour chaque valeur
3. Choisir la valeur qui converge le plus vite sans diverger

```python
# Tester différents learning rates
learning_rates = [0.001, 0.01, 0.1, 1.0]
best_alpha = None
best_J = float('inf')

for alpha in learning_rates:
    theta = np.zeros(2)
    _, J_history = gradient_descent(X_b, y, theta, alpha, 1000)
    final_J = J_history[-1]
    print(f"α={alpha:.3f}, J_final={final_J:.4f}")
    
    if final_J < best_J:
        best_J = final_J
        best_alpha = alpha

print(f"\nMeilleur α: {best_alpha}")
```

---

## 5.2 Régularisation

### Pourquoi régulariser ?
Éviter l'**overfitting** (le modèle mémorise les données d'entraînement au lieu d'apprendre des patterns généraux).

```python
def compute_cost_regularized(X, y, theta, lambda_reg):
    """
    Fonction de coût avec régularisation L2
    
    J(θ) = (1/(2m)) * Σ(h_θ(x) - y)^2 + (λ/(2m)) * Σθ_j^2 (pour j ≥ 1)
    """
    m = len(y)
    predictions = X @ theta
    errors = predictions - y
    J = (1 / (2 * m)) * np.sum(errors**2)
    
    # Régularisation (sans le bias θ₀)
    reg_term = (lambda_reg / (2 * m)) * np.sum(theta[1:]**2)
    J += reg_term
    
    return J

def gradient_descent_regularized(X, y, theta, alpha, num_iters, lambda_reg):
    """
    Descente de gradient avec régularisation
    """
    m = len(y)
    J_history = []
    
    for _ in range(num_iters):
        predictions = X @ theta
        errors = predictions - y
        
        # Gradient sans régularisation
        gradient = (1 / m) * (X.T @ errors)
        
        # Ajouter la régularisation (sans le bias)
        gradient[1:] += (lambda_reg / m) * theta[1:]
        
        theta = theta - alpha * gradient
        J_history.append(compute_cost_regularized(X, y, theta, lambda_reg))
    
    return theta, J_history
```

**Formules de Régularisation L2** :

Fonction de coût :
$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2
$$

Mise à jour des poids :
$$
\theta_j := \theta_j - \alpha \left( \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} \theta_j \right) \quad \text{pour } j \geq 1
$$

**Remarques** :
- **λ (lambda)** : Contrôle l'importance de la régularisation
  - λ = 0 → Pas de régularisation
  - λ grand → Plus de régularisation (modèle plus simple)
- **Ne pas régulariser θ₀** (le bias) → `theta[1:]` dans le code
- **Effet** : Réduit la magnitude des poids → modèle moins sensible aux variations des données

---

### 📈 Détection de l'Overfitting

**Graphique typique** :
- **Overfitting** : Coût d'entraînement **très bas**, coût de validation **élevé**
- **Bon ajustement** : Coût d'entraînement et de validation **proches et bas**
- **Underfitting** : Coût d'entraînement et de validation **élevés**

**Solutions** :
| Problème | Solution |
|----------|----------|
| Overfitting | Augmenter λ, ajouter plus de données, réduire le nombre de features |
| Underfitting | Réduire λ, ajouter des features, augmenter la complexité du modèle |
| Coût élevé | Continuer l'entraînement, augmenter α |

---

# 6. 📊 Évaluation et Diagnostics

> **Objectif** : Évaluer les performances du modèle et diagnostiquer les problèmes.

## 6.1 Métriques d'Évaluation

```python
def predict(theta, X):
    """
    Prédit la classe (0 ou 1) pour la régression logistique
    """
    prob = sigmoid(X @ theta)
    return (prob >= 0.5).astype(int)

def compute_accuracy(y_true, y_pred):
    """
    Calcule l'accuracy (pourcentage de bonnes prédictions)
    """
    return np.mean(y_true == y_pred) * 100

def confusion_matrix(y_true, y_pred):
    """
    Calcule la matrice de confusion
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    return np.array([[tn, fp], [fn, tp]])
```

**Métriques Clés** :

1. **Accuracy** :
   $$
   \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} \times 100\%
   $$

2. **Precision** :
   $$
   \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
   $$

3. **Recall (Sensibilité)** :
   $$
   \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
   $$

4. **F1-Score** :
   $$
   \text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
   $$

**Matrice de Confusion** :

| | Prédit 0 | Prédit 1 |
|-----------|----------|----------|
| **Vrai 0** | TN | FP |
| **Vrai 1** | FN | TP |

Où :
- **TP** = Vrais Positifs
- **TN** = Vrais Négatifs
- **FP** = Faux Positifs (Erreur de type I)
- **FN** = Faux Négatifs (Erreur de type II)

```python
# Exemple d'évaluation

# Prédire sur les données de test
y_pred = predict(theta_opt, X_test_b)

# Calculer les métriques
accuracy = compute_accuracy(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}%")
print(f"Matrice de confusion:\n{cm}")

# Calculer Precision, Recall, F1
tp, fp, fn, tn = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
```

---

## 6.2 Learning Curves

### Pourquoi tracer les learning curves ?
Pour diagnostiquer si le modèle souffre de **bias élevé** (underfitting) ou de **variance élevée** (overfitting).

```python
def plot_learning_curve(X, y, theta, alpha, num_iters):
    """
    Trace la learning curve (coût en fonction du nombre d'exemples)
    """
    m = len(y)
    train_errors = []
    val_errors = []
    
    for i in range(1, m + 1):
        # Entraîner sur les i premiers exemples
        X_train = X[:i]
        y_train = y[:i]
        
        # Calculer le coût d'entraînement
        train_cost = compute_cost(X_train, y_train, theta)
        train_errors.append(train_cost)
        
        # Calculer le coût de validation (sur tous les exemples restants)
        if i < m:
            X_val = X[i:]
            y_val = y[i:]
            val_cost = compute_cost(X_val, y_val, theta)
            val_errors.append(val_cost)
    
    # Tracer
    plt.plot(range(1, m+1), train_errors, label='Train')
    plt.plot(range(1, m+1), val_errors + [val_errors[-1]], label='Validation')
    plt.xlabel('Nombre d\'exemples')
    plt.ylabel('Erreur')
    plt.legend()
    plt.title('Learning Curve')
    plt.show()
```

### 📈 Interprétation des Learning Curves

**Cas 1 : Bias Élevé (Underfitting)** :
- Les erreurs d'entraînement et de validation sont **toutes les deux élevées**
- Le modèle est **trop simple** pour capturer les patterns des données
- **Solutions** :
  - Ajouter plus de features (ex: features polynomiaux)
  - Réduire la régularisation (diminuer λ)
  - Utiliser un modèle plus complexe (ex: réseau de neurones au lieu de régression linéaire)

**Cas 2 : Variance Élevée (Overfitting)** :
- L'erreur d'entraînement est **basse**, mais l'erreur de validation est **élevée**
- Le modèle **mémorise** les données d'entraînement
- **Solutions** :
  - Ajouter plus de données d'entraînement
  - Augmenter la régularisation (augmenter λ)
  - Réduire le nombre de features

**Cas 3 : Bon Ajustement** :
- Les erreurs d'entraînement et de validation **convergent vers une valeur basse**
- Le modèle généralise bien

---

# 7. 🔍 Études de Cas (Exemples Concrets)

> **Objectif** : Appliquer les concepts à des exemples réels tirés de tes TP.

## 7.1 TP1 : Régression Linéaire sur ex1data1.txt

**Problème** : Prédire les profits d'un food truck en fonction de la population d'une ville.

**Données** :
- `ex1data1.txt` : 2 colonnes (population, profit)
- m = 97 exemples

**Solution complète** :

```python
# 1. Charger les données
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, 0]
y = data[:, 1]
m = len(y)

# 2. Ajouter le terme constant
X_b = np.c_[np.ones((m, 1)), X]

# 3. Initialiser theta
theta = np.zeros(2)

# 4. Descente de gradient
alpha = 0.01
num_iters = 1500
theta_opt, J_history = gradient_descent(X_b, y, theta, alpha, num_iters)

# 5. Visualiser les résultats
plt.scatter(X, y, color='red', marker='x', label='Données')
plt.plot(X, X_b @ theta_opt, color='blue', label='Régression linéaire')
plt.xlabel('Population de la ville (en 10,000s)')
plt.ylabel('Profit (en $10,000s)')
plt.legend()
plt.title('Régression Linéaire - TP1')
plt.show()

# 6. Prédire pour une nouvelle ville
population = 35000  # 35,000 habitants
x_new = np.array([1, population/10000])  # Normaliser comme les données d'entraînement
profit_pred = x_new @ theta_opt
print(f"Profit prédit pour {population} habitants: ${profit_pred*10000:,.2f}")
```

**Questions d'examen typiques pour TP1** :

1. **Quelle est la dimension de X et de y ?**
   - `X.shape = (97,)` (vecteur de 97 populations)
   - `y.shape = (97,)` (vecteur de 97 profits)
   - `X_b.shape = (97, 2)` (après ajout du bias)

2. **Pourquoi normaliser les données ?**
   - Dans ce cas, la population varie de ~5 à ~25 (en 10,000s), donc pas besoin de normaliser
   - Mais si les features avaient des échelles très différentes, il faudrait normaliser

3. **Comment choisir α ?**
   - Essayer α = 0.01, 0.03, 0.1 → α = 0.01 fonctionne bien

4. **Interprétation de θ₀ et θ₁** :
   - θ₀ = profit de base quand population = 0
   - θ₁ = augmentation du profit par 10,000 habitants supplémentaires

---

## 7.2 TP2 : Régression Logistique sur ex2data1.txt

**Problème** : Prédire l'admission à l'université en fonction des scores à deux examens.

**Données** :
- `ex2data1.txt` : 3 colonnes (score examen 1, score examen 2, admission)
- m = 100 exemples
- y ∈ {0, 1} (0 = non admis, 1 = admis)

```python
# 1. Charger les données
data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]
m = len(y)

# 2. Normaliser les données
X_norm, mu, sigma = feature_normalize(X)

# 3. Ajouter le terme constant
X_b = np.c_[np.ones((m, 1)), X_norm]

# 4. Initialiser theta
theta = np.zeros(3)

# 5. Descente de gradient
alpha = 0.1
num_iters = 400
theta_opt, J_history = gradient_descent_logistic(X_b, y, theta, alpha, num_iters)

# 6. Tracer la frontière de décision
plt.scatter(X_norm[y==1, 0], X_norm[y==1, 1], color='red', marker='*', label='Admis')
plt.scatter(X_norm[y==0, 0], X_norm[y==0, 1], color='yellow', edgecolor='black', marker='o', label='Non admis')

# Créer une grille
x1_min, x1_max = X_norm[:, 0].min(), X_norm[:, 0].max()
x2_min, x2_max = X_norm[:, 1].min(), X_norm[:, 1].max()
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))

# Prédire pour chaque point de la grille
X_grid = np.c_[np.ones((100*100, 1)), xx1.ravel(), xx2.ravel()]
preds = (sigmoid(X_grid @ theta_opt) >= 0.5).astype(int)
preds = preds.reshape(xx1.shape)

plt.contourf(xx1, xx2, preds, alpha=0.2, colors=['yellow', 'red'])
plt.xlabel('Score Examen 1 (normalisé)')
plt.ylabel('Score Examen 2 (normalisé)')
plt.legend()
plt.title('Frontière de décision - TP2')
plt.show()

# 7. Évaluer le modèle
y_pred = predict(theta_opt, X_b)
accuracy = compute_accuracy(y, y_pred)
print(f"Accuracy: {accuracy:.2f}%")
```

**Questions d'examen typiques pour TP2** :

1. **Quelle est la dimension de X, y, θ ?**
   - `X.shape = (100, 2)` (100 exemples, 2 features)
   - `y.shape = (100,)` (100 labels)
   - `theta.shape = (3,)` (θ₀ + θ₁ + θ₂)
   - `X_b.shape = (100, 3)` (après ajout du bias)

2. **Pourquoi normaliser les données ?**
   - Les scores des examens peuvent avoir des échelles différentes → normalisation nécessaire

3. **Interprétation de la frontière de décision** :
   - La frontière est une **ligne droite** car c'est une régression logistique linéaire
   - Si la frontière n'est pas linéaire, il faudrait ajouter des features polynomiaux

4. **Comment calculer l'accuracy ?**
   - `accuracy = (nombre de bonnes prédictions) / (nombre total d'exemples) * 100`

---

## 7.3 TP3 : Réseau de Neurones sur ex3data1.mat

**Problème** : Classifier des chiffres manuscrits (0-9) à partir d'images 20x20 pixels.

**Données** :
- `ex3data1.mat` : images de chiffres (5000 exemples)
- X : (5000, 400) (20x20 pixels aplatis)
- y : (5000,) (chiffres 0-9)

```python
# 1. Charger les données
data = loadmat('ex3data1.mat')
X = data['X']  # (5000, 400)
y = data['y']  # (5000,)

# 2. Convertir y en one-hot encoding
y_onehot = np.eye(10)[y.astype(int).flatten()]  # (5000, 10)

# 3. Initialiser les poids
input_size = 400
hidden_size = 25
output_size = 10
Theta1, Theta2 = initialize_weights(input_size, hidden_size, output_size)

# 4. Entraîner le réseau
Theta1, Theta2, J_history = train_nn(
    X, y_onehot, input_size=400, hidden_size=25, output_size=10,
    alpha=0.1, num_iters=100, lambda_reg=0.1
)

# 5. Tracer la courbe de coût
plt.plot(J_history)
plt.xlabel('Itérations')
plt.ylabel('J(Θ)')
plt.title('Convergence - TP3')
plt.show()

# 6. Prédire sur un exemple
def predict_nn(X, Theta1, Theta2):
    a3, _, _, _ = forward_propagation(X, Theta1, Theta2)
    return np.argmax(a3, axis=1)

# Prédire le premier exemple
sample_idx = 0
pred = predict_nn(X[sample_idx:sample_idx+1], Theta1, Theta2)
print(f"Vrai label: {y[sample_idx]}, Prédiction: {pred[0]}")
```

**Questions d'examen typiques pour TP3** :

1. **Quelle est la dimension de Θ¹ et Θ² ?**
   - `Theta1.shape = (25, 401)` (25 neurones cachés, 400 features + 1 bias)
   - `Theta2.shape = (10, 26)` (10 neurones de sortie, 25 neurones cachés + 1 bias)

2. **Pourquoi utiliser le one-hot encoding ?**
   - Pour convertir les labels (0-9) en vecteurs binaires (ex: 3 → [0,0,0,1,0,0,0,0,0,0])
   - Permet d'utiliser la cross-entropy pour la classification multi-classe

3. **Pourquoi initialiser les poids aléatoirement ?**
   - Pour briser la symétrie et permettre à chaque neurone d'apprendre des features différentes

4. **Interprétation de la courbe de coût** :
   - Si J(Θ) **décroît** → le réseau apprend
   - Si J(Θ) **stagne** → essayer un learning rate plus grand ou plus d'itérations
   - Si J(Θ) **oscille** → réduire le learning rate

---

# 8. ❓ Questions Fréquentes en Examen

> **Objectif** : Répondre aux questions théoriques et pratiques courantes.

## 8.1 Questions sur les Dimensions

### Q: Quelle est la dimension de X et de y dans la régression linéaire ?
**R:**
- `X.shape = (m, n)` où m = nombre d'exemples, n = nombre de features
- `y.shape = (m,)` (vecteur de m éléments)
- Après ajout du bias : `X_b.shape = (m, n+1)`
- `theta.shape = (n+1,)`

**Exemple TP1** :
- `X.shape = (97,)` → `X_b.shape = (97, 2)`
- `theta.shape = (2,)`

---

### Q: Quelle est la dimension de Θ¹ et Θ² dans un réseau de neurones ?
**R:**
- `Theta1.shape = (s₂, s₁ + 1)` où s₁ = taille couche entrée, s₂ = taille couche cachée
- `Theta2.shape = (s₃, s₂ + 1)` où s₃ = taille couche sortie
- **+1** pour le terme de bias

**Exemple TP3** :
- s₁ = 400 (pixels), s₂ = 25 (neurones cachés), s₃ = 10 (chiffres)
- `Theta1.shape = (25, 401)`
- `Theta2.shape = (10, 26)`

---

## 8.2 Questions sur la Normalisation

### Q: Quand faut-il normaliser les données ?
**R:**
✅ **Oui, normaliser** si :
- Les features ont des **échelles très différentes** (ex: âge en années vs. revenu en euros)
- Tu utilises la **descente de gradient** (convergence plus rapide)
- Tu utilises la **régression logistique** ou des **réseaux de neurones**

❌ **Non, ne pas normaliser** si :
- Les features sont déjà à la même échelle (ex: scores entre 0 et 100)
- Tu utilises un modèle **insensible à l'échelle** (ex: arbre de décision)

---

### Q: Comment normaliser manuellement ?
**R:**
```python
mu = np.mean(X, axis=0)      # Moyenne par feature
sigma = np.std(X, axis=0)    # Écart-type par feature
X_norm = (X - mu) / sigma     # Normalisation
```
**Formule** : $x_{norm} = \frac{x - \mu}{\sigma}$

---

## 8.3 Questions sur le Learning Rate

### Q: Comment choisir le learning rate (α) ?
**R:**
1. **Essayer plusieurs valeurs** : 0.001, 0.01, 0.1, 1.0
2. **Tracer J(θ)** pour chaque valeur
3. **Choisir la valeur qui** :
   - **Converge le plus vite** (J(θ) décroît rapidement)
   - **Ne diverge pas** (J(θ) ne va pas vers l'infini)

**Symptômes** :
- **α trop grand** → J(θ) **oscille** ou **explose** → réduire α
- **α trop petit** → J(θ) **décroît très lentement** → augmenter α
- **α bon** → J(θ) **décroît rapidement et converge**

---

### Q: Pourquoi α ne doit pas être trop grand ?
**R:**
Si α est trop grand, la descente de gradient peut **"sauter"** par-dessus le minimum :

```
J(θ)
  ^
  |       /\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n  |      /  \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n  |     /    \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n  |    /      \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n  |   /        \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n  |  /          \\\\\\\\\\\\\\\\\\\\\\\\\\\\n  | /            \\\\\\\\\\\\\\\\\\\\\\\\\
  |/              \\\\\\\\\\\\\\\\\\\\\\\n  +----------------> θ
       α trop grand : sauts trop grands → divergence
       α bon : pas trop grands → convergence
```

---

## 8.4 Questions sur la Régularisation

### Q: Pourquoi utiliser la régularisation ?
**R:**
Pour **éviter l'overfitting** :
- Sans régularisation, le modèle peut **mémoriser** les données d'entraînement
- Avec régularisation, les poids sont **plus petits** → modèle plus simple et généralisable

**Analogie** :
- Sans régularisation = étudiant qui **apprend par cœur** sans comprendre
- Avec régularisation = étudiant qui **comprend les concepts**

---

### Q: Comment choisir λ (lambda) ?
**R:**
1. **Essayer plusieurs valeurs** : 0, 0.01, 0.1, 1, 10
2. **Tracer les learning curves** pour chaque λ
3. **Choisir λ qui** :
   - **Réduit l'erreur de validation** (évite l'overfitting)
   - **Ne fait pas trop augmenter l'erreur d'entraînement** (évite l'underfitting)

**Règle empirique** :
- λ = 0 → Pas de régularisation
- λ = 0.1 → Régularisation légère
- λ = 1 → Régularisation modérée
- λ = 10 → Régularisation forte

---

## 8.5 Questions sur les Graphiques

### Q: Comment interpréter une courbe de coût qui oscille ?
**R:**
- **Cause** : Learning rate (α) **trop grand**
- **Solution** : **Réduire α** (ex: passer de 1.0 à 0.1 ou 0.01)

---

### Q: Comment interpréter une courbe de coût qui stagne ?
**R:**
- **Cause 1** : Learning rate (α) **trop petit** → augmenter α
- **Cause 2** : **Minimum local** → essayer une autre initialisation
- **Cause 3** : **Pas assez d'itérations** → augmenter `num_iters`

---

### Q: Comment interpréter une frontière de décision ?
**R:**
- **Ligne droite** → Régression logistique linéaire
- **Courbe complexe** → Régression logistique avec features polynomiaux
- **Frontière mal placée** → Le modèle n'a pas convergé ou α est mal choisie
- **Frontière trop complexe** → Overfitting → augmenter λ ou ajouter des données

---

## 8.6 Questions sur le Code

### Q: Pourquoi utiliser `X @ theta` au lieu de `np.dot(X, theta)` ?
**R:**
- `@` est l'**opérateur de produit matriciel** introduit dans Python 3.5+
- Plus **lisible** et **concise**
- **Équivalent** à `np.dot(X, theta)` pour les tableaux NumPy

**Exemple** :
```python
# Ces deux lignes font la même chose
result1 = X @ theta
result2 = np.dot(X, theta)
```

---

### Q: Pourquoi `theta[1:]` dans la régularisation ?
**R:**
- Le **bias (θ₀)** ne doit **pas être régularisé**
- `theta[0]` = θ₀ (bias)
- `theta[1:]` = θ₁, θ₂, ..., θₙ (poids des features)
- **Formule** : $J(θ) = ... + \frac{λ}{2m} \sum_{j=1}^{n} θ_j^2$ (j commence à 1, pas 0)

---

## 8.7 Questions sur les Réseaux de Neurones

### Q: Pourquoi ne pas initialiser les poids à zéro ?
**R:**
Si tous les poids sont à zéro :
- Tous les neurones de la **même couche** auront la **même valeur**
- La **backpropagation** donnera les **mêmes gradients** pour tous les neurones
- Les poids **resteront égaux** pendant tout l'entraînement
- Le réseau **ne peut pas apprendre** de features différentes

**Solution** : Initialiser **aléatoirement** pour briser la symétrie

---

### Q: Quelle est la différence entre forward et backward propagation ?
**R:**

| **Forward Propagation** | **Backward Propagation** |
|------------------------|---------------------------|
| Calcul des prédictions | Calcul des gradients |
| Entrée → Sortie | Sortie → Entrée |
| Utilise les poids | Met à jour les poids |
| Pas de gradients | Utilise les erreurs |

**Analogie** :
- Forward = **calculer** une réponse
- Backward = **apprendre** de ses erreurs

---

# 🎯 Résumé des Étapes Clés pour l'Examen

## ✅ Checklist avant de rendre ton notebook

1. **Chargement des données** :
   - [ ] Vérifier le format du fichier (.txt, .mat, .csv)
   - [ ] Utiliser la bonne fonction de chargement (`np.loadtxt`, `loadmat`, `pd.read_csv`)
   - [ ] Séparer X et y correctement
   - [ ] Vérifier les dimensions avec `.shape`

2. **Prétraitement** :
   - [ ] Ajouter le terme constant (bias) si nécessaire
   - [ ] Normaliser les données si les features ont des échelles différentes
   - [ ] Découper en train/validation/test

3. **Initialisation** :
   - [ ] Initialiser θ à zéro pour la régression linéaire/logistique
   - [ ] Initialiser aléatoirement pour les réseaux de neurones
   - [ ] Ne pas oublier le bias dans les matrices de poids

4. **Entraînement** :
   - [ ] Choisir un learning rate approprié (tester plusieurs valeurs)
   - [ ] Choisir un nombre d'itérations suffisant
   - [ ] Ajouter la régularisation si nécessaire
   - [ ] Tracer la courbe de coût pour vérifier la convergence

5. **Évaluation** :
   - [ ] Calculer l'accuracy sur les données de test
   - [ ] Tracer la matrice de confusion si classification
   - [ ] Interpréter les résultats

6. **Vérifications finales** :
   - [ ] Relire le code pour éviter les erreurs de dimensions
   - [ ] Vérifier que J(θ) décroît bien
   - [ ] S'assurer que les prédictions ont du sens

---

## 📚 Formules à Retenir

### Régression Linéaire
- Hypothèse : $h_\theta(x) = \theta^T x$
- Coût : $J(\theta) = \frac{1}{2m}\sum(h_\theta(x^{(i)}) - y^{(i)})^2$
- Gradient : $\nabla J(\theta) = \frac{1}{m}X^T(X\theta - y)$

### Régression Logistique
- Sigmoïde : $g(z) = \frac{1}{1+e^{-z}}$
- Hypothèse : $h_\theta(x) = g(\theta^T x)$
- Coût : $J(\theta) = -\frac{1}{m}\sum[y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))]$
- Gradient : $\nabla J(\theta) = \frac{1}{m}X^T(h_\theta(X) - y)$

### Réseaux de Neurones
- Forward : $a^{(l+1)} = g(\Theta^{(l)} a^{(l)})$
- Coût : $J(\Theta) = -\frac{1}{m}\sum\sum y_k \log(h_k) + \frac{\lambda}{2m}\sum\sum (\Theta_{ji}^{(l)})^2$
- Backward : $\delta^{(l)} = (\Theta^{(l)})^T \delta^{(l+1)} \odot g'(z^{(l)})$

### Régularisation
- Coût : $J(\theta) = J_{original} + \frac{\lambda}{2m}\sum_{j=1}^{n} \theta_j^2$
- Gradient : $\frac{\partial J}{\partial \theta_j} = \frac{\partial J_{original}}{\partial \theta_j} + \frac{\lambda}{m}\theta_j$ (pour j ≥ 1)

---

## 💡 Derniers Conseils

- **Lis bien les questions** : Parfois la réponse est dans l'énoncé !
- **Commence par le plus simple** : Régression linéaire avant les réseaux de neurones
- **Vérifie les dimensions** : 90% des erreurs viennent de dimensions incompatibles
- **Trace des graphiques** : Ça aide à comprendre ce qui ne va pas
- **Gère ton temps** : Ne passe pas trop de temps sur une seule question
- **Relis-toi** : Avant de rendre, vérifie que tout est cohérent

**BON EXAMEN ! 🚀**

---

## 📦 Cellule d'Import à Exécuter au Début

```python
# Cellule d'imports à exécuter au début de ton examen
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import optimize  # Pour les méthodes d'optimisation avancées

# Configuration de matplotlib pour de meilleurs graphiques
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (10, 6)
```