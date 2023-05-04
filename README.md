Download Link: https://assignmentchef.com/product/solved-csci5521-homework-1
<br>
<ol>

 <li>Consider doing least squares regression based on a training set Z<sub>train </sub>= {(<em>x<sup>t</sup>,r<sup>t</sup></em>)<em>,t </em>= 1<em>,…,N</em>}, where <em>x<sup>t </sup></em>∈ R and <em>r<sup>t </sup></em>∈ R.</li>

</ol>

(i) Consider fitting a linear model of the form

<em>g</em><sub>1</sub>(<em>x</em>) = <em>w</em><sub>1</sub><em>x </em>+ <em>w</em><sub>0 </sub><em>,</em>

with unknown parameters <em>w</em><sub>1</sub><em>,w</em><sub>0 </sub>∈ R, which are selected so as to minimize the following empirical loss:

<em>N</em>

<em> .</em>

<em>t</em>=1

Derive the optimal values of (<em>w</em><sub>1</sub><em>,w</em><sub>0</sub>) clearly showing all steps of the derivation. (ii) Consider fitting a polynomial model of the form

<em>g</em>2(<em>x</em>) = <em>v</em>2<em>x</em>2020 + <em>v</em>1<em>x </em>+ <em>v</em>0 <em>,</em>

with unknown parameters <em>v</em><sub>2</sub><em>,v</em><sub>1</sub><em>,v</em><sub>0 </sub>∈ R, which are selected so as to minimize the following empirical loss:

<em>N</em>

<em> .</em>

<em>t</em>=1

Derive the optimal values of <em>v</em><sub>2</sub><em>,v</em><sub>1</sub><em>,v</em><sub>0 </sub>clearly showing all steps of the derivation.<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>

(iii) For a given training set Z<sub>train</sub>, let () be the optimal values of (<em>w</em><sub>1</sub><em>,w</em><sub>0</sub>) in (i) above, and let () be the optimal values of (<em>v</em><sub>2</sub><em>,v</em><sub>1</sub><em>,v</em><sub>0</sub>) in (ii) above. Professor Gopher claims that the following is true for any given Z<sub>train</sub>:

Is Professor Gopher’s claim correct? Clearly explain your answer.<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>

<ol start="2">

 <li>Consider the following 4 × 4 matrix:</li>

</ol>

<sup></sup>1    1     1      1 <sup></sup>

<em>A </em>= <sub></sub>1    2     4      8 <sub> </sub><em>.</em>

<sub></sub>1    3     9     27<sub></sub>

1    4    16    64

<ul>

 <li>What are the values of tr(<em>A</em>)<em>,</em>tr(<em>A<sup>T</sup></em>)<em>,</em>tr(<em>A<sup>T</sup>A</em>), and tr(<em>AA<sup>T</sup></em>).<sup>3</sup></li>

 <li>From a geometric perspective, explain how the absolute value of |<em>A</em>| (determinant of <em>A</em>) can be computed.</li>

 <li>Are the rows of <em>A </em>linearly independent? Clearly explain your answer.<sup>2</sup></li>

</ul>

(For this problem, you can use python libraries to arrive at your answer. If you do that, clearly explain what you did and why. There is a way to arrive at the answer without using python libraries.)

<strong>Programming assignments: </strong>The next two problems involve programming. We will be considering three datasets (derived from two available datasets) for these assignments:

<ul>

 <li>Boston: The Boston housing dataset comes pre-packaged with scikit-learn. The dataset has 506 points, 13 features, and 1 target (response) variable. You can find more information about the dataset here:</li>

</ul>

https://github.com/rupakc/UCI-Data-Analysis/tree/master/Boston Housing Dataset/Boston Housing

While the original dataset is for a regression problem, we will create two classification datasets for the homework. Note that you only need to work with the response <em>r </em>to create these classification datasets.

<ol start="50">

 <li>Boston50: Let <em>τ</em><sub>50 </sub>be the median (50th percentile) over all <em>r </em>(response) values. Create a 2-class classification problem such that <em>y </em>= 1 if <em>r </em>≥ <em>τ</em><sub>50 </sub>and <em>y </em>= 0 if <em>r &lt; τ</em><sub>50</sub>. By construction, note that the class priors will be.</li>

 <li>Boston75: Let <em>τ</em><sub>75 </sub>be the 75th percentile over all <em>r </em>(response) values. Create a 2-class classification problem such that <em>y </em>= 1 if <em>r </em>≥ <em>τ</em><sub>75 </sub>and <em>y </em>= 0 if <em>r &lt; τ</em><sub>75</sub>. By construction, note that the class priors will be.</li>

</ol>

<ul>

 <li>Digits: The Digits dataset comes prepackaged with scikit-learn. The dataset has 1797 points, 64 features, and 10 classes corresponding to ten numbers 0<em>,</em>1<em>,…,</em> The dataset was (likely) created from the following dataset:</li>

</ul>

http://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits

The 2-class classification datasets from Boston50, Boston75, and the 10-class classification dataset from Digits will be used in the following two problems.

<ol start="3">

 <li>We will consider three methods from scikit-learn: LinearSVC, SVC, and LogisticRegression. Use the following parameters for the different methods mentioned:</li>

</ol>

LinearSVC: max iter=2000

SVC: gamma=‘scale’, C=10

LogisticRegression: penalty=‘l2’, solver=‘lbfgs’, multi class=‘multinomial’, max iter=5000

(i) Develop code for my cross val(method,<em>X</em>,<strong>y</strong>,<em>k</em>), which performs <em>k</em>-fold crossvalidation on (<em>X,</em><strong>y</strong>) using method, and returns the error rate in each fold. Using my cross val, report the error rates in each fold as well as the mean and standard deviation of error rates across folds for the three methods: LinearSVC, SVC, and LogisticRegression, applied to the three classification datasets: Boston50, Boston75, and Digits.

You will have to submit (a) <strong>code </strong>and (b) <strong>summary of results </strong>for my cross val:

<ul>

 <li><strong>Code</strong>: You will have to submit code for my cross val(method,<em>X</em>,<strong>y</strong>,<em>k</em>) (main file) as well as a wrapper code q3i().</li>

</ul>

<strong>The main file </strong>has <strong>input</strong>: (1) method, which specifies the (class) name of one of the three classification methods under consideration, (2) <em>X</em>,<strong>y</strong>, which is data for the 2-class or 10-class classification problem, (3) <em>k</em>, the number of folds for crossvalidation, and <strong>output</strong>: (1) the test set error rates for each of the <em>k </em>folds.

<strong>The wrapper code </strong>has no input and is used to prepare the datasets, and make calls to my cross val(method,<em>X</em>,<strong>y</strong>,<em>k</em>) to generate the results for each dataset and each method. Make sure the calls to my cross val(method,<em>X</em>,<strong>y</strong>,<em>k</em>) are made in the following order and add a print to the terminal before each call to show which method and dataset is being used:

<ol>

 <li>LinearSVC with Boston50; 2. LinearSVC with Boston75; 3. LinearSVC with</li>

</ol>

Digits,

<ol start="4">

 <li>SVC with Boston50; 5. SVC with Boston75; 6. SVC with Digits,</li>

 <li>LogisticRegression with Boston50; 8. LogisticRegression with Boston75;</li>

 <li>LogisticRegression with Digits.</li>

</ol>

For example, the first call to my cross val(method,<em>X</em>,<strong>y</strong>,<em>k</em>) with <em>k </em>= 10 should result in the following output:

Error rates for LinearSVC with Boston50:

Fold 1: ###

Fold 2: ###

…

Fold 10: ###

Mean: ###

Standard Deviation: ###

<ul>

 <li><strong>Summary of results</strong>: For each dataset and each method, report the test set error rates for each of the <em>k </em>= 10 folds, the mean error rate over the <em>k </em>folds, and the standard deviation of the error rates over the <em>k </em> Make a table to present the results for each method and each dataset (9 tables in total). Include a column in the table for each fold, and add two columns at the end to show the overall mean error rate and standard deviation over the <em>k </em>folds. For example:</li>

</ul>

<table width="421">

 <tbody>

  <tr>

   <td width="33"> </td>

   <td width="33"> </td>

   <td colspan="9" width="320">Error rates for LinearSVC with Boston50</td>

   <td width="35"> </td>

  </tr>

  <tr>

   <td width="33">F1</td>

   <td width="33">F2</td>

   <td width="33">F3</td>

   <td width="33">F4</td>

   <td width="33">F5</td>

   <td width="33">F6</td>

   <td width="33">F7</td>

   <td width="33">F8</td>

   <td width="33">F9</td>

   <td width="40">F10</td>

   <td width="51">Mean</td>

   <td width="35">SD</td>

  </tr>

  <tr>

   <td width="33">#</td>

   <td width="33">#</td>

   <td width="33">#</td>

   <td width="33">#</td>

   <td width="33">#</td>

   <td width="33">#</td>

   <td width="33">#</td>

   <td width="33">#</td>

   <td width="33">#</td>

   <td width="40">#</td>

   <td width="51">#</td>

   <td width="35">#</td>

  </tr>

 </tbody>

</table>

(ii) Develop code for my train test(method,<em>X</em>,<strong>y</strong>,<em>π</em>,<em>k</em>), which performs random splits on the data (<em>X,</em><strong>y</strong>) so that <em>π </em>∈ [0<em>,</em>1] fraction of the data is used for training using method, rest is used for testing, and the process is repeated <em>k </em>times, after which the code returns the error rate for each such train-test split. Using my train test, with <em>π </em>= 0<em>.</em>75 and <em>k </em>= 10, report the mean and standard deviation of error rate for the three methods: LinearSVC, SVC, and LogisticRegression, applied to the three classification datasets: Boston50, Boston75, and Digits.

You will have to submit (a) <strong>code </strong>and (b) <strong>summary of results </strong>for my train test:

(a) <strong>Code</strong>: You will have to submit code for my train test(method,<em>X</em>,<strong>y</strong>,<em>π</em>,<em>k</em>) (main file) as well as a wrapper code q3ii().

<strong>This main file </strong>has <strong>input</strong>: (1) method, which specifies the (class) name of one

of the three classification methods under consideration, (2) <em>X</em>,<strong>y</strong>, which is data for the 2-class or 10-class classification problem, (3) <em>π</em>, the fraction of data chosen randomly to be used for training, (4) <em>k</em>, the number of times the train-test split will be repeated, and <strong>output</strong>: (1) the test set error rates for each of the <em>k </em>folds printed to the terminal.

<strong>The wrapper code </strong>has no input and is used to prepare the datasets, and make calls to my train test(method,<em>X</em>,<strong>y</strong>,<em>π</em>,<em>k</em>) to generate the results for each dataset and each method (9 combinations in total). Make sure the calls to my train test(method,<em>X</em>,<strong>y</strong>,<em>π</em>,<em>k</em>) are made in the following order and add a print to the terminal before each call to show which method and dataset is being used:

<ol>

 <li>LinearSVC with Boston50; 2. LinearSVC with Boston75; 3. LinearSVC with</li>

</ol>

Digits,

<ol start="4">

 <li>SVC with Boston50; 5. SVC with Boston75; 6. SVC with Digits,</li>

 <li>LogisticRegression with Boston50; 8. LogisticRegression with Boston75;</li>

 <li>LogisticRegression with Digits.</li>

</ol>

(b) <strong>Summary of results</strong>: For each dataset and each method, report the test set error rates for each of the <em>k </em>= 10 runs with <em>π </em>= 0<em>.</em>75, the mean error rate over the <em>k </em>folds, and the standard deviation of the error rates over the <em>k </em>folds. Make a table to present the results for each method and each dataset (9 tables in total). Include a column in the table for each run, and add two columns at the end to show the overall mean error rate and standard deviation over the <em>k </em>runs.

<ol start="4">

 <li>The problem considers a preliminary exercise in ‘feature engineering’ with focus on the Digits dataset. Represented as (<em>X,</em><strong>y</strong>), the Digits dataset has <em>X </em>∈ R<sup>1797×6<a href="#_ftn3" name="_ftnref3">[3]</a></sup>, i.e., 1797 training points, each having 64 features, and <strong>y </strong>∈ {0<em>,</em>1<em>,…,</em>9}<sup>1797</sup>, i.e., 1797 training labels with each <em>y<sub>i </sub></em>∈ {0<em>,</em>1<em>,…,</em>9}. We will consider three methods from scikit-learn: LinearSVC, SVC, and LogisticRegression for this problem. Use the following parameters for the different methods mentioned:</li>

</ol>

LinearSVC: max iter=2000

SVC: gamma=‘scale’, C=10

LogisticRegression: penalty=‘l2’, solver=‘lbfgs’, multi class=‘multinomial’, max iter=5000

<ul>

 <li>For the Digits dataset, starting with <em>X </em>∈ R<sup>1797×64</sup>, you will create a new feature representation <em>X</em><sup>˜</sup><sub>1 </sub>∈ R<sup>1797×32 </sup>as follows: Construct a (random) matrix <em>G </em>∈ R<sup>64×32 </sup>where each element <em>g<sub>ij </sub></em>∼ <em>N</em>(0<em>,</em>1), i.e., sampled independently from a univariate normal distribution, and then compute <em>X</em><sup>˜</sup><sub>1 </sub>= <em>XG</em>. Using (<em>X</em><sup>˜</sup><sub>1</sub><em>,</em><strong>y</strong>), perform 10-fold crossvalidation<sup>4 </sup>using the three methods: LinearSVC, SVC, and LogisticRegression, and report the mean and the standard deviation of the 10-fold test set error rate.<a href="#_ftn4" name="_ftnref4"><sup>[4]</sup></a> The creation of <em>X</em><sup>˜</sup><sub>1 </sub>will be done based on a function rand proj(<em>X,d</em>), where <em>d </em>= 32 for this problem, and the function will return <em>X</em><sup>˜</sup><sub>1</sub>.</li>

 <li>For the Digits dataset, starting with <em>X </em>∈ R<sup>1797×64</sup>, you will create a new feature representation <em>X</em><sup>˜</sup><sub>2 </sub>∈ R<sup>1797×2144 </sup>as follows: For any training data <strong>x</strong><em><sub>i </sub></em>∈ R<sup>64</sup>, let the elements be <em>x<sub>ij</sub>,j </em>= 1<em>,…,</em> The new feature set ˜<em>x<sub>i </sub></em>∈ R<sup>2144 </sup>will include all the original features <em>x<sub>ij</sub>,j </em>= 1<em>,…,</em>64, squares of the original features <em>x</em><sup>2</sup><em><sub>ij</sub>,j </em>= 1<em>,…,</em>64, and products of all the original features <em>x<sub>ij</sub>x<sub>ij</sub></em>0<em>,j &lt; j</em><sup>0</sup><em>,j </em>= 1<em>,…,</em>64<em>,j</em><sup>0 </sup>= <em>j</em>+1<em>,…,</em>64. You should verify that the new ˜<em>x<sub>i </sub></em>∈ R<sup>2144 </sup>and hence <em>X</em><sup>˜</sup><sub>2 </sub>∈ R<sup>1797×2144</sup>. Using (<em>X</em><sup>˜</sup><sub>2</sub><em>,</em><strong>y</strong>), perform 10-fold cross-validation<sup>4 </sup>using the three methods: LinearSVC, SVC, and LogisticRegression, and report the mean and the standard deviation of the 10-fold test set error rate. The creation of <em>X</em><sup>˜</sup><sub>2 </sub>will be done based on a function quad proj(<em>X</em>), and the function will return <em>X</em><sup>˜</sup><sub>2</sub>.</li>

</ul>

You will have to submit (a) <strong>code </strong>and (b) <strong>summary of results </strong>for all three parts:

<ul>

 <li><strong>Code</strong>: You will have to submit code for rand proj(<em>X</em>,<em>d</em>), quad proj(<em>X</em>) as well as a wrapper code q4().</li>

</ul>

rand proj<strong>(</strong><em>X</em><strong>,</strong><em>d</em><strong>) </strong>has <strong>input</strong>: (1) <em>X</em>, which is data (features) for the classification problem, (2) <em>d</em>, the dimensionality of the projected features, and <strong>output</strong>: (1) <em>X</em><sup>˜ </sup>∈ R<sup>1797×<em>d</em></sup>, the new data for the problem. This output array does not need to be printed to the terminal. quad proj<strong>(</strong><em>X</em><strong>) </strong>has <strong>input</strong>: <em>X</em>, which is data (features) for the classification problem, and <strong>output</strong>: (1) <em>X</em>˜<sub>2</sub>, the new data with all linear and quadratic combinations of features as described above. This output array does not need to be printed to the terminal.

<strong>The wrapper code </strong>has no input and uses these above functions to execute all the classification exercises outlined in (i) and (ii) above and print the test set error rates for each of the <em>k </em>folds to the terminal. Make sure the exercises are executed in the following order and add a print to the terminal before each execution to show which method and dataset is being used:

<ol>

 <li>LinearSVC with <em>X</em><sup>˜</sup><sub>1</sub>; 2. LinearSVC with <em>X</em><sup>˜</sup><sub>2</sub>,</li>

 <li>SVC with <em>X</em><sup>˜</sup><sub>1</sub>; 4. SVC with <em>X</em><sup>˜</sup><sub>2</sub>,</li>

 <li>LogisticRegression with <em>X</em><sup>˜</sup><sub>1</sub>; 6. LogisticRegression with <em>X</em><sup>˜</sup><sub>2</sub>.</li>

</ol>

<ul>

 <li><strong>Summary of results</strong>: For each dataset, i.e., <em>X</em><sup>˜</sup><sub>1 </sub>and <em>X</em><sup>˜</sup><sub>2</sub>, and each method, report the mean error rate over the <em>k </em>folds, and the standard deviation of the error rates over the <em>k </em> Make a table to present the results for each method and each dataset (6 tables in total). Include a column in the table for each fold, and add two columns at the end to show the overall mean error rate and standard deviation over the <em>k </em>folds.</li>

</ul>

<a href="#_ftnref1" name="_ftn1">[1]</a> It is ok to leave the solution in terms of a linear system, say <em>A</em><strong>v </strong>= <strong>b</strong>, where <em>A</em>∈R<sup>3×3</sup>, <strong>b</strong>∈R<sup>3 </sup>are known, and <strong>v </strong>= [<em>v</em><sub>1 </sub><em>v</em><sub>2 </sub><em>v</em><sub>2</sub>]<em><sup>T </sup></em><sup>∈</sup>R<sup>3 </sup>is a vector of the unknown parameters. If you choose to do this, please also mention your preferred approach to solve such a linear system.

<a href="#_ftnref2" name="_ftn2">[2]</a> A correct answer with insufficient or incorrect explanation will not get any credit. <sup>3</sup>For this problem, you can use python libraries for the computations.

<a href="#_ftnref3" name="_ftn3">[3]</a> Please use your own code my cross val for this problem.

<a href="#_ftnref4" name="_ftn4">[4]</a> Since <em>G </em>is a random matrix, every time you generate <em>G </em>and repeat the procedure, your results will be a bit different.