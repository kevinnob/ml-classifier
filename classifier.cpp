#include "csvstream.hpp"
#include <iostream>
#include <string>
#include <map>
#include <cmath>
#include <set>
#include <sstream>

using namespace std;

// A Classifier represents a text classification model trained using
// a Multi-Variate Bernoulli Naive Bayes algorithm.
// It learns label and word statistics from training data, and can
// predict the label of unseen posts.
class Classifier{
    public:
        // REQUIRES: train must be a valid csvstream with "tag" and "content" columns.
        // MODIFIES: cout, total_posts, unique_words, label_count_posts,
        //           word_count_posts, label_word_count_posts
        // EFFECTS:  Reads all posts from the training stream and updates internal
        //           statistics (word counts, label counts, vocabulary). If print is
        //           true, prints each training example and vocabulary size.
        void train(csvstream& train , bool print){
            map<string, string> row;
            if(print){cout << "training data:" << endl;}

            while(train >> row){
                string label = row["tag"];
                string text = row["content"];
                set<string> words = get_unique_words(text);
                total_posts++;
                if(print){
                    cout << "  label = " << label << ", content = " << text << endl;
                }
                

                for(auto it = words.begin(); it != words.end(); it++){
                    unique_words.insert(*it);
                    label_word_count_posts[label][*it]++;
          
                    word_count_posts[*it]++;
                }
                label_count_posts[label]++;
            }
            cout << "trained on " << total_posts << " examples" << endl;

            if(print){
                cout << "vocabulary size = " << unique_words.size() << endl;
                
            }
            cout << endl;
        }
        // REQUIRES: train() must have been called successfully before calling this.
        //           test must be a valid csvstream with "tag" and "content" columns.
        // MODIFIES: cout
        // EFFECTS:  Reads all posts from the test stream, predicts their labels using
        //           the trained model, and prints prediction results. Outputs the
        //           number of correct predictions and overall accuracy.
        void prediction(csvstream& test) {
            map<string, string> row;
            int correct = 0;
            int total = 0;
            

            cout << "test data:" << endl;
        
            while(test >> row){
                string actual = row["tag"];
                string text = row["content"];

                set<string> words = get_unique_words(text);
                map<string, double> log_prob_scores = compute_log_score(words);
                pair<string, double> predicted = get_higest_prob(log_prob_scores);

                cout << "  correct = " << actual
                << ", predicted = " << predicted.first
                << ", log-probability score = " << predicted.second << endl;
                cout << "  content = " << text << endl;
                cout << endl;

                if(predicted.first == actual){
                    correct++;
                }
                total++;
            }

            cout << "performance: " << correct << " / " << total
            << " posts predicted correctly" <<endl;
        }
        // REQUIRES: train() must have been called successfully before calling this.
        // MODIFIES: cout
        // EFFECTS:  Prints the learned model parameters including log-priors for each
        //           class and log-likelihoods for each label-word pair.
        void print() const{

            cout << "classes:" << endl;
            for (const auto &p : label_count_posts){
                double logPrior= log(p.second / (double) total_posts);
                cout << "  " << p.first << ", " << p.second
                << " examples, log-prior = " << logPrior << endl;

            }

            cout << "classifier parameters:" << endl;
            for(const auto &l : label_word_count_posts){
                for(const auto & w : l.second){
                    double logLikelihood = log(w.second / 
                        (double) label_count_posts.at(l.first));
                    cout << "  " << l.first << ":" << w.first
                    << ", count = " << w.second
                    << ", log-likelihood = " << logLikelihood << endl;
                }

            }
            cout << endl;
        }

    private:
        int total_posts = 0;
        set<string> unique_words;
        map<string, int> word_count_posts;
        map<string, int> label_count_posts;
        map<string, map<string, int>> label_word_count_posts;

        // REQUIRES: None
        // MODIFIES: None
        // EFFECTS:  Returns a set of unique, whitespace-delimited words appearing in str.
        set<string> get_unique_words(const string &str) {
            istringstream source(str);
            set<string> words;
            string word;
            while (source >> word) {
                words.insert(word);
            }
            return words;
        }

        // REQUIRES: train() must have been called successfully before calling this.
        // MODIFIES: None
        // EFFECTS:  Computes and returns a map from each label to its log-probability
        //           score for classifying a post containing the given words.
        map<string, double> compute_log_score(const set<string> &testWords){
            map<string, double> prob_labels;

            for(const auto &p : label_count_posts){
                string label = p.first;
                double prob = log(p.second / (double) total_posts);

                for(const string &word : testWords){
                    if(label_word_count_posts.count(label) && 
                    label_word_count_posts.at(label).count(word)){
                        prob += log(label_word_count_posts.at(label).at(word)/
                            (double) label_count_posts.at(label));
                    }
                    else if(unique_words.count(word)){
                        prob += log(word_count_posts.at(word)/(double) total_posts);
                    }
                    else{
                        prob += log(1.0/total_posts);
                    }
                }
                prob_labels[label] = prob;
            }
            return prob_labels;

        }

        // REQUIRES: probs must be non-empty.
        // MODIFIES: None
        // EFFECTS:  Returns the (label, score) pair with the highest probability.
        //           Breaks ties alphabetically by label.
        pair<string, double> get_higest_prob(const map<string, double> &probs) const{
            pair<string, double> highest = {probs.begin()->first, probs.begin()->second};

            for(auto it = probs.begin(); it != probs.end(); it++){
                if((it->second > highest.second) || (it->second == highest.second 
                    && it->first < highest.first)){
                    highest = {it->first, it->second};
                }
            }
            return highest;
        }
};


// REQUIRES: argc is 2 or 3. argv[1] must be a valid CSV file path.
//           If argc is 3, argv[2] must be a valid CSV file path.
// MODIFIES: cout
// EFFECTS:  Trains a classifier on the training file. If only one file is provided,
//           prints the trained model parameters. If two files are provided,
//           uses the trained model to predict labels for the test file and
//           reports performance. Returns 0 on success, 1 on error.
int main(int argc, char* argv[]){
    Classifier c;
    cout.precision(3);

    if(argc != 2 && argc != 3){
        cout << "Usage: classifier.exe TRAIN_FILE [TEST_FILE]" << endl;
        return 1;
    }
    if(argc == 2){
        try{
            csvstream csvtrain(argv[1]);
            c.train(csvtrain, true);
            c.print();
        } catch(const csvstream_exception& e){
            cout << "Error opening file: " << argv[1] << endl;
            return 1;
        }
    }

    if(argc == 3 ){
        try{
            csvstream csvtrain(argv[1]);
            c.train(csvtrain, false);
        } catch(const csvstream_exception& e){
            cout << "Error opening file: " << argv[1] << endl;
            return 1;
        }

        try{
            csvstream csvTest(argv[2]);
            c.prediction(csvTest);
        } catch(const csvstream_exception& e){
            cout << "Error opening file: " << argv[2] << endl;
            return 1;
        }
    }

    return 0;
}