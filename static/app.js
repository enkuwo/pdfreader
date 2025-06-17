let chartLoaded = false;

function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.getElementById('tab-' + tabName).classList.add('active');

    if (tabName === 'chart' && !chartLoaded) {
        renderChart();
        chartLoaded = true;
    }
}

function searchSummary() {
    const input = document.getElementById("searchInput").value.toLowerCase();
    const summaryDiv = document.getElementById("tab-summary");
    const original = summaryDiv.innerText;

    if (!input) {
        summaryDiv.innerHTML = summaryDiv.innerText;
        return;
    }

    const regex = new RegExp(`(${input})`, "gi");
    const highlighted = original.replace(regex, "<mark>$1</mark>");
    summaryDiv.innerHTML = highlighted;
}

function renderChart() {
    const ctx = document.getElementById('myChart');
    if (ctx) {
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['example', 'data', 'result'],
                datasets: [{
                    label: 'Keyword Count',
                    data: [5, 3, 2],
                    backgroundColor: ['#007bff', '#6c757d', '#28a745']
                }]
            }
        });
    }
}

let currentQuestion = 0;
let score = 0;
let answered = []; // To track if each question has been answered

function renderQuestion() {
    if (!quizData || !quizData.length || !quizData[currentQuestion]) {
        document.getElementById('quiz-area').innerHTML = "<p>No quiz data available.</p>";
        document.getElementById('prev-btn').style.display = "none";
        document.getElementById('next-btn').style.display = "none";
        return;
    }

    // If quiz is over, show the result
    if (currentQuestion >= quizData.length) {
        showFinalScore();
        return;
    }

    const q = quizData[currentQuestion];
    let html = `<div class="question"><strong>Q${currentQuestion+1}: ${q.question}</strong></div>`;
    html += '<div class="answers">';
    q.choices.forEach((choice, idx) => {
        html += `<button class="answer-btn" onclick="checkAnswer(event, '${choice.replace(/'/g, "\\'")}', this)">${choice}</button>`;
    });
    html += '</div><div class="feedback" id="feedback"></div>';
    document.getElementById('quiz-area').innerHTML = html;

    // Show/hide navigation buttons
    document.getElementById('prev-btn').style.display = currentQuestion > 0 ? "inline-block" : "none";
    document.getElementById('next-btn').style.display = currentQuestion < quizData.length - 1 ? "inline-block" : "none";
}

function checkAnswer(event, selected, btn) {
    event.preventDefault();
    const q = quizData[currentQuestion];
    const buttons = document.querySelectorAll('.answer-btn');
    // Only allow scoring the first time
    if (!answered[currentQuestion]) {
        if (selected === q.answer) {
            score++;
        }
        answered[currentQuestion] = true;
    }
    buttons.forEach(btn => {
        btn.disabled = true;
        if (btn.textContent === q.answer) {
            btn.classList.add('correct');
        }
        if (btn.textContent === selected && btn.textContent !== q.answer) {
            btn.classList.add('incorrect');
        }
    });
    const feedback = document.getElementById('feedback');
    if (selected === q.answer) {
        feedback.textContent = "✅ Correct!";
        feedback.style.color = "#38d39f";
    } else {
        feedback.textContent = `❌ Incorrect. The correct answer is "${q.answer}".`;
        feedback.style.color = "#f87171";
    }
}

function showFinalScore() {
    document.getElementById('quiz-area').innerHTML = `
        <h3>Your score: ${score} / ${quizData.length}</h3>
        <button class="btn" onclick="restartQuiz()">Retry Quiz</button>
        <form action="/" method="get" style="display:inline;">
            <button type="submit" class="btn">Back to Upload</button>
        </form>
    `;
    document.getElementById('prev-btn').style.display = "none";
    document.getElementById('next-btn').style.display = "none";
}

function restartQuiz() {
    currentQuestion = 0;
    score = 0;
    answered = [];
    renderQuestion();
}
function toggleDarkMode() {
    document.body.classList.toggle("dark");
    localStorage.setItem("darkMode", document.body.classList.contains("dark"));
}
window.onload = () => {
    if (localStorage.getItem("darkMode") === "true") {
        document.body.classList.add("dark");
    }
};

// Navigation button handlers
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('prev-btn').addEventListener('click', function() {
        if (currentQuestion > 0) {
            currentQuestion--;
            renderQuestion();
        }
    });
    document.getElementById('next-btn').addEventListener('click', function() {
        if (currentQuestion < quizData.length - 1) {
            currentQuestion++;
            renderQuestion();
        } else {
            currentQuestion++;
            renderQuestion();
        }
    });
    renderQuestion();
});
