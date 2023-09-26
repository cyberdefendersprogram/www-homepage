const quizData = {
    "quiz_questions": [
      {
        "topic": "cloud observability",
        "question": "Which of the following is a primary component of cloud observability?",
        "options": [
          "Access Management",
          "Log Aggregation",
          "Firewall Configuration",
          "Encryption Settings"
        ],
        "correct_answer": "b",
        "explanation": "Cloud observability primarily consists of three components: metrics, logs, and traces. Log aggregation is an essential process where logs from various sources are collected and united in one place, facilitating centralized analysis and monitoring."
      },
      {
        "topic": "cloud observability",
        "question": "AWS CloudTrail primarily facilitates the tracking of what kind of information in an AWS environment?",
        "options": [
          "Virtual Machine Metrics",
          "User and API activity",
          "Network Traffic Volume",
          "Database Query Performance"
        ],
        "correct_answer": "b",
        "explanation": "AWS CloudTrail is a service that enables governance, compliance, operational auditing, and risk auditing of your AWS account. It mainly helps you to log, continuously monitor, and retain account activity related to actions across your AWS infrastructure, which includes tracking user and API activity."
      },
      {
        "topic": "cloud observability",
        "question": "In the context of cloud observability, what does the term 'traceability' typically refer to?",
        "options": [
          "The ability to track changes to infrastructure settings",
          "The ability to follow a transaction or workflow through various components of a system",
          "Tracing the origins of data inputs",
          "Documenting system downtime periods"
        ],
        "correct_answer": "b",
        "explanation": "Traceability in the realm of cloud observability refers to the ability to trace a transaction or workflow as it propagates through the components of a system. This is critical for diagnosing performance bottlenecks and understanding system behavior."
      },
      {
        "topic": "cloud observability",
        "question": "What is a key benefit of utilizing AWS CloudTrail in a cloud computing environment?",
        "options": [
          "Enhancing the graphical representation of system architectures",
          "Optimizing the cloud costs",
          "Enabling the monitoring and logging of account activity",
          "Accelerating the deployment of new features"
        ],
        "correct_answer": "c",
        "explanation": "AWS CloudTrail allows you to monitor and log account activity related to actions across your AWS infrastructure. This is vital for governance, risk auditing, and operational auditing, helping organizations to maintain a secure and compliant environment."
      },
      {
        "topic": "cloud observability",
        "question": "Within the realm of cloud observability, what role does anomaly detection play?",
        "options": [
          "Identifying patterns in user behavior",
          "Recognizing deviations from established baselines or norms",
          "Assisting in the optimization of resource allocation",
          "Facilitating the encryption of data transmissions"
        ],
        "correct_answer": "b",
        "explanation": "Anomaly detection in cloud observability is centered on identifying patterns or occurrences that deviate from established baselines or norms. Recognizing these anomalies is crucial for early detection of potential issues or security breaches, allowing for timely intervention."
      }
    ]
  };
  
  
  let currentQuestionIndex = 0;
  let currentTopic = '';

  const card = document.getElementById('quiz-card');
  const cardFront = document.getElementById('card-front');
  const questionText = document.getElementById('question-text');
  const optionsList = document.getElementById('options-list');
  const answerText = document.getElementById('answer-text');
  const explanationText = document.getElementById('explanation-text');
  const nextButton = document.getElementById('next-button');
  const topicSelect = document.getElementById('topic-select');

  function populateTopics() {
    const topics = [...new Set(quizData.quiz_questions.map(q => q.topic))];
    console.log(topics);
    topicSelect.innerHTML = topics.map(topic => `<option value="${topic}">${topic}</option>`).join('');
    currentTopic = topics[0];
  }

  function loadQuestion() {
    const questionsByTopic = quizData.quiz_questions.filter(q => q.topic === currentTopic);
    const questionData = questionsByTopic[currentQuestionIndex];

    questionText.textContent = questionData.question;
    optionsList.innerHTML = questionData.options.map((option, idx) => 
      `<li>${String.fromCharCode(97 + idx)}. ${option}</li>`
    ).join('');
    answerText.textContent = 'Answer: ' + questionData.correct_answer;
    explanationText.textContent = 'Explanation: ' + questionData.explanation;
  }

  function goToNextQuestion() {
    const questionsByTopic = quizData.quiz_questions.filter(q => q.topic === currentTopic);
    currentQuestionIndex = (currentQuestionIndex + 1) % questionsByTopic.length;
    loadQuestion();
    card.style.transform = 'rotateY(0deg)';
  }
  
  nextButton.addEventListener('click', (event) => {
    event.stopPropagation();
    goToNextQuestion();
  });

  topicSelect.addEventListener('change', (event) => {
    currentTopic = event.target.value;
    currentQuestionIndex = 0;
    loadQuestion();
  });

  populateTopics();
  loadQuestion();
  document.addEventListener('DOMContentLoaded', (event) => {
    cardFront.addEventListener('click', () => {
        card.style.transform = 'rotateY(180deg)';
      });
  });