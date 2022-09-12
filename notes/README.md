# Guidelines for Scribe Duties

## General Guidelines
For 6.S965 class, each student is required to scribe for a few lectures. The scribe duties cover 5% of final grade.

### Preparation
Please [fork](https://help.github.com/articles/fork-a-repo/)  the course repository. All your notes will be first pushed to your fork of the repository.

### Write up lecture notes
- Write up and add your notes to `notes/` in your fork of the repository. Make sure the file name of your notes is formatted as `lecture-<dd>.md`. For example, the note for lecture 02 should be readable via `notes/lecture-02.md`.
- The scribe notes should be written in Markdown.  If you are not familiar with markdown, please spend 5 min looking at this [tutorial](https://commonmark.org/help/tutorial/index.html).
- Place all images used in the notes to the the lecture folder under the `notes/figures` folder. For example, all images for note `lecture-02.md` should be placed in `notes/figures/lecture-02/` folder.
- Make sure the title of the notes is formatted as `Lecture <dd>: <lecture title>`. For example, `Lecture 01: Introduction to TinyML and Efficient Deep Learning`.
- Make sure the lecturer and the authors/scribes information is correct.

### Submission
The due date for each note is 1 week after the lecture. To submit the notes for review, please do as follows:

 1. Create  [a pull request](https://help.github.com/articles/about-pull-requests/)  (PR) into the master branch of the course repository.
 2. Make sure you title your pull request  `Lecture <dd>: <lecture title>`.
 3. TAs will review your notes and approve or request changes. Please update your notes according to TAs' requests.
 4. Once TAs are satisfied with the quality of the scribed notes, your PR will be merged.
 5. Once the PR is merged, authors of the submitted notes get credit.

## Tips

### Equations
Rendering LaTeX mathematical expressions in inline and display modes is supported by using [KaTeX](https://khan.github.io/KaTeX/).

- **inline mode**: simply surround your math expression with `$`. For example, fully-connected layer computes $\mathbf{Y}=\mathbf{X}\mathbf{W}^{T}$, where inputs $\mathbf{X}$ is a tensor of shape $(n, c_i)$, weights $\mathbf{W}$ is a tensor of shape $(c_o, c_i)$, and $\mathbf{X}$ is a tensor of shape $(n, c_o)$ .
- **display mode**: simply surround your math expression with `$$` and place it as a separate paragraph. For example,

$$\texttt{MACs}_{\texttt{conv}} = c_o \cdot c_i \cdot k_h \cdot k_w \cdot h_o \cdot w_o$$

> You can find more information about **LaTeX** mathematical expressions [here](http://meta.math.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference).

### Code Blocks


### Citations
