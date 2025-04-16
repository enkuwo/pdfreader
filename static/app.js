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
