function analyze() {
    let review = document.getElementById("review").value;

    fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({review: review})
    })
    .then(res => res.json())
    .then(data => {
        let result = document.getElementById("result");

        result.innerHTML = data.result;

        if (data.result === "Genuine") {
            result.style.background = "green";
        } else {
            result.style.background = "red";
        }
    });
}

// Load charts only on dashboard page
if (document.getElementById("barChart")) {

    fetch("/stats")
    .then(res => res.json())
    .then(data => {

        let labels = Object.keys(data);
        let values = Object.values(data);

        new Chart(document.getElementById("barChart"), {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{ data: values }]
            }
        });

        new Chart(document.getElementById("pieChart"), {
            type: "pie",
            data: {
                labels: labels,
                datasets: [{ data: values }]
            }
        });

    });
} 