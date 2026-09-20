# LaTeX Math Code Reference

A cheatsheet of LaTeX math syntax for use in research write-ups, publications, and any future MDX/markdown content on the site (e.g. `components/pages/research`).

## Delimiters

| Context | Syntax | Renders as |
|---|---|---|
| Inline math | `$E = mc^2$` | $E = mc^2$ |
| Display (block) math | `$$E = mc^2$$` | centered, own line |
| LaTeX native inline | `\( ... \)` | same as `$ ... $` |
| LaTeX native block | `\[ ... \]` | same as `$$ ... $$` |
| Numbered equation (LaTeX doc) | `\begin{equation} ... \end{equation}` | auto-numbered |

## Core Symbols

```latex
\alpha \beta \gamma \delta \epsilon \theta \lambda \mu \pi \sigma \phi \omega
\Delta \Sigma \Omega \Phi \Lambda
\infty \partial \nabla \forall \exists \in \notin \subset \subseteq
\leq \geq \neq \approx \equiv \pm \times \div \cdot
\rightarrow \leftarrow \Rightarrow \Leftrightarrow \mapsto
```

## Fractions, Roots, Powers

```latex
\frac{a}{b}
\sqrt{x}
\sqrt[n]{x}
x^{n}
x_{i}
x_{i}^{n}
```

## Sums, Products, Integrals, Limits

```latex
\sum_{i=1}^{n} x_i
\prod_{i=1}^{n} x_i
\int_{a}^{b} f(x)\,dx
\lim_{x \to \infty} f(x)
```

## Matrices and Vectors

```latex
\begin{bmatrix} a & b \\ c & d \end{bmatrix}
\begin{pmatrix} a & b \\ c & d \end{pmatrix}
\vec{v}
\mathbf{v}
```

## Sets, Logic, Probability

```latex
\{ x \in \mathbb{R} \mid x > 0 \}
P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}
\binom{n}{k}
```

## Common Number Sets

```latex
\mathbb{N}  % naturals
\mathbb{Z}  % integers
\mathbb{Q}  % rationals
\mathbb{R}  % reals
\mathbb{C}  % complex
```

## Aligning Multi-line Equations

```latex
\begin{align}
a &= b + c \\
  &= d + e
\end{align}
```

## Example Snippets

Quadratic formula:

```latex
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
```

Bayes' theorem:

```latex
P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}
```

Shannon entropy:

```latex
H(X) = -\sum_{i=1}^{n} p(x_i) \log_2 p(x_i)
```

Big-O notation:

```latex
f(n) = O(g(n)) \iff \exists\, c, n_0 > 0 : f(n) \leq c\,g(n)\ \forall n \geq n_0
```

## Rendering Note

This repo does not currently ship a LaTeX renderer (no KaTeX/MathJax/`remark-math` dependency — see `package.json`). These snippets are raw LaTeX source for reference/authoring only; they will not render as typeset math anywhere in the app until a renderer is added.
