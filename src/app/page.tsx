"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Check, Sparkles, RotateCcw } from "lucide-react";
import { Card, CardContent } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Progress } from "../components/ui/progress";
import confetti from "canvas-confetti";

const descriptions = [
  "Install Anaconda / VS Code, Set up Jupyter Notebook, Write Python scripts",
  "Lists, Dictionaries, Functions in Python, Exercises on Kaggle",
  "NumPy basics: arrays, slicing, operations",
  "Pandas basics: DataFrames, loading and filtering CSV",
  "Matplotlib & Seaborn: Visualize data",
  "Linear Algebra basics, 3Blue1Brown videos",
  "Probability and Statistics: mean, variance",
  "Supervised learning, Linear Regression, scikit-learn",
  "House price predictor hands-on (Kaggle)",
  "Logistic Regression, Titanic dataset (Kaggle)",
  "Evaluation Metrics: Accuracy, Precision, Recall, F1",
  "Train/Test Split, Cross-Validation, Pipelines",
  "Overfitting/Underfitting, Regularization",
  "Push Titanic project to GitHub",
  "Decision Trees, Random Forest theory",
  "Random Forest hands-on: hyperparameter tuning",
  "KNN theory and hands-on",
  "SVM theory, train SVM on iris dataset",
  "GridSearchCV, cross-validation",
  "Start ML Project 1: Dataset + Scope",
  "Train multiple models + evaluate",
  "K-Means Clustering theory",
  "Hands-on: Cluster Iris dataset",
  "PCA theory + hands-on on digits dataset",
  "t-SNE & DBSCAN, visualize data",
  "Customer segmentation or image compression",
  "Document unsupervised project on GitHub",
  "ML Summary & Concept Review",
  "Neural Networks: Neurons, weights, activations",
  "Feedforward Net in NumPy",
  "Keras Sequential API, Train MNIST model",
  "Loss functions, optimizers, activations",
  "Gradient Descent & Backpropagation",
  "Experiment with learning rate, epochs",
  "Push digit recognizer to GitHub",
  "CNN layers, filters, pooling",
  "Build CIFAR-10 CNN classifier",
  "Data augmentation + regularization",
  "Transfer learning with ResNet/VGG",
  "Image Classifier App using Streamlit",
  "Deploy model on Render",
  "Write blog about classifier project",
  "Tokenization, embeddings, text classification",
  "RNNs/LSTM Theory + Examples",
  "IMDB Sentiment Analysis with LSTM",
  "Text preprocessing: stopwords, stemming",
  "Evaluate: confusion matrix, accuracy",
  "Push NLP project to GitHub",
  "Watch Transformer intuition by Jay Alammar",
  "BERT & Transformer architecture",
  "Text classification using HuggingFace pipeline",
  "QnA using DistilBERT",
  "Text generation with GPT-2",
  "Train custom model with HuggingFace Trainer",
  "Document Transformer project on GitHub",
  "Blog recap + LinkedIn update",
  "LLMs and Retrieval Augmented Generation (RAG)",
  "LangChain basics with OpenAI API",
  "Vector embeddings: OpenAI + ChromaDB",
  "QnA chatbot over PDF",
  "Add Streamlit interface to chatbot",
  "Test & Deploy LangChain app",
  "Write case study on GitHub/LinkedIn",
  "Choose capstone project & plan architecture",
  "Capstone: Resume bot / News summarizer / Tutor bot",
  "Build full-stack app with FastAPI + Streamlit",
  "Deploy & document capstone project",
  "Save models with Pickle/ONNX",
  "Serve model with FastAPI endpoints",
  "Dockerize model + app",
  "Deploy on Azure/Render",
  "Model monitoring basics",
  "Polish repo for resume",
  "Write blog on deployment journey",
  "Create GitHub profile README",
  "Clean & organize project repos",
  "Write blogs for LinkedIn/Medium",
  "Create PDF and Notion Portfolio",
  "Apply to jobs, contribute to OSS, prep interviews",
  "Apply to jobs, contribute to OSS, prep interviews",
  "Apply to jobs, contribute to OSS, prep interviews"
];

const days = Array.from({ length: 84 }, (_, i) => `Day ${i + 1}`);

export default function Home() {
  const [completed, setCompleted] = useState<number[]>([]);
  const [xp, setXP] = useState(0);
  const [level, setLevel] = useState(1);
  const [quote, setQuote] = useState("");
  const [showTrivia, setShowTrivia] = useState(true);

  const quotes = [
    "Keep going, you're doing great!",
    "One step closer to mastery!",
    "Persistence beats talent when talent doesn't persist!",
    "Today's effort is tomorrow's success!",
    "Code a little every day and you’ll be unstoppable!",
    "Small progress is still progress!"
  ];

  const trivia = [
    "The term 'artificial intelligence' was coined in 1956 at Dartmouth College.",
    "Deep learning models are inspired by how the human brain processes data.",
    "GPT stands for Generative Pre-trained Transformer.",
    "ImageNet is one of the largest visual databases used in AI research.",
    "Reinforcement learning is based on the reward/punishment principle."
  ];

  useEffect(() => {
    setQuote(quotes[Math.floor(Math.random() * quotes.length)]);
  }, [completed]);

  useEffect(() => {
    const saved = localStorage.getItem("aiChecklist");
    if (saved) {
      const parsed = JSON.parse(saved);
      setCompleted(parsed.completed || []);
      setXP(parsed.xp || 0);
      setLevel(parsed.level || 1);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(
      "aiChecklist",
      JSON.stringify({ completed, xp, level })
    );
  }, [completed, xp, level]);

  const toggleDay = (index: number) => {
    const alreadyDone = completed.includes(index);
    const updatedCompleted = alreadyDone
      ? completed.filter((i) => i !== index)
      : [...completed, index];

    setCompleted(updatedCompleted);

    if (!alreadyDone) {
      const newXP = xp + 10;
      setXP(newXP);
      if (newXP >= level * 100) {
        setLevel((prev) => prev + 1);
        setXP(0);
      }

      if ((index + 1) % 7 === 0) {
        confetti({
          particleCount: 100,
          spread: 70,
          origin: { y: 0.6 }
        });
      }
    }
  };

  const resetProgress = () => {
    if (confirm("Are you sure you want to reset your progress?")) {
      setCompleted([]);
      setXP(0);
      setLevel(1);
      localStorage.removeItem("aiChecklist");
    }
  };

  const [triviaQuote, setTriviaQuote] = useState("");

useEffect(() => {
  setTriviaQuote(trivia[Math.floor(Math.random() * trivia.length)]);
}, []);


  return (
    <div className="p-6 space-y-6">
      <div className="text-center space-y-2">
        <h1 className="text-4xl font-bold">🧠 AI Mastery Streak</h1>
        <div className="text-sm text-muted">Level {level} | XP: {xp}/100</div>
        <div className="italic text-blue-500">{quote}</div>
        <div className="flex justify-center items-center gap-4 mt-2">
          <Button variant="outline" size="sm" onClick={() => setShowTrivia(!showTrivia)}>
            {showTrivia ? "Hide Trivia" : "Show Trivia"}
          </Button>
          <Button variant="destructive" size="sm" onClick={resetProgress}>
            <RotateCcw className="w-4 h-4 me-2" /> Reset Progress
          </Button>
        </div>
        {triviaQuote && <div className="text-sm text-gray-600 mt-2">💡 {triviaQuote}</div>}
      </div>

      <Progress value={(completed.length / days.length) * 100} className="h-4" />

      <div className="text-center text-xs text-gray-500">
        {completed.length} of {days.length} days completed
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
        {days.map((day, index) => (
          <motion.div
            key={index}
            whileHover={{ scale: 1.02 }}
            className={`rounded-3xl shadow border p-4 transition-all duration-300 ${
              completed.includes(index)
                ? "border-green-500 bg-green-100/70"
                : "border-gray-300 bg-white"
            }`}
          >
            <Card className="bg-transparent">
              <CardContent className="flex flex-col items-center justify-between gap-3">
                <h2 className="text-lg font-semibold">{day}</h2>
                <p className="text-sm text-gray-600 text-center">{descriptions[index]}</p>
                <Button
                  onClick={() => toggleDay(index)}
                  variant={completed.includes(index) ? "secondary" : "default"}
                  className="w-full"
                >
                  {completed.includes(index) ? <><Check className="me-2" /> Done</> : "Mark as Done"}
                </Button>
                {completed.includes(index) && index % 7 === 6 && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="text-yellow-600 flex items-center gap-2"
                  >
                    <Sparkles className="w-4 h-4" />
                    <span>🔥 1-Week Streak!</span>
                  </motion.div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
