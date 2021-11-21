<strong>Preliminary Analysis</strong> <br>
  <li><strong>Data Cleaning Code </strong> <br>
        Code for cleaning and processing your data. Include a data dictionary for your transformed dataset. 
    <ul> 
      <li>
      Data Dictionary:
        <ul>
            <li><strong>id</strong>: A unique identifier for each tweet</li>
            <li><strong>text</strong>: the text of the tweet</li>
            <li><strong>location</strong>: the location the tweet was sent</li>
            <li><strong>keyword</strong>: a particular keyword from the tweet</li>
            <li><strong>target</strong>: denotes whether a tweet is about a real disaster(1) or not(0)</li>
        </ul>
      </li> 
    </ul> 
  </li>
  <li><strong>Exploratory Analysis </strong> <br>
       Describe what work you have done so far and include the code. This may include descriptive statistics, graphs and charts, and preliminary models. 
    <ul> 
      <li>
        For this preliminary analysis, we noticed that there were inconsistencies in the data columns so we adjusted them using the steps in the next bullet point.
        <li>
          The <strong>keyword</strong> column did not extract all the disaster keywords from the text. So we extracted the keyword column and text column and applied a function to extract the correct keywords by word tokenizing and extracting the words which intersect with the disaster word vector. 
        <li>
          The same was done for the <strong>location</strong> column.
        </li>
        </li>
      </li> 
    </ul> 
  </li>
  <li><strong>Challenges </strong> <br>
        Describe any challenges you've encountered so far. Let me know if there's anything you need help with! 
    <ul> 
      <li>
        There were many challenges when attempting to correct the inconsistent data columns. 
      </li> 
      <li>
          The <strong>keyword</strong> column wasn't as difficult but still came as a challenge as attempting the fillna using normal means wasnt working for us. But, we figured it out. 
      </li> 
      <li>
        The <strong>location</strong> column was difficult. Hands down took the longest to implement and even now doesnt work correctly as there arent efficient and free NER libraries to extract 'GPE' from texts. We attempted using 3 different types, one of which included using <strong>spaCy</strong>, but that didnt work with Google Colab. The other two that we used were very time consuming to run. Took a whomping <strong>17~20 minutes</strong> when using on the train data set. 
      </li> 
    </ul> 
  </li>
  <li><strong>Future Work </strong> <br>
        Describe what work you are planning to complete for the final analysis.
    <ul> 
      <li>Future work involves using our cleaned data and features as input for models suited for classification, like Naive Bayes and Logisitic Regression and training these models</li> 
      <li>Making predictions off our trained models and evaluating performance with accuracy scores and confusion matrices</li>
      <li>Defining our grid of hyperparameter values and using GridSearchCV() to systematically find the best peforming model</li>
      </ul> 
  </li>
  <li><strong>Contributions </strong> <br>
        Describe the contributions that each group member made. 
    <ul> 
      <li>
      Daniel Hernandez
      <ul> 
        <li>Researched and acquired the datasets
        </li>
        <li>Helped in brainstorming. 
        </li> 
        <li>Created and organized juypter notebook
        </li>
        <li>Implemented visualizations of data for preliminary analysis
        </li>
        <li>Assisted in thinking of procedure to clean data columns
        </li>
    </ul> 
      </li> 
      <li>
      Justin Park
      <ul> 
        <li>Helped in brainstorming.
        </li>
        <li>Implemented functions to clean keyword and location columns.
        </li> 
        <li>Implemented procedure to clean data columns.
        </li>
    </ul> 
      </li> 
    </ul> 
  </li>
</ul>

