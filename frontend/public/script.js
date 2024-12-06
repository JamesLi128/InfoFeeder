console.log("Script loaded");

document.getElementById('input-form').addEventListener('submit', function(event) {
  event.preventDefault();
  const userInput = document.getElementById('user-input').value;
  console.log('User Input:', userInput);

  // Send the data to the backend
  fetch('http://127.0.0.1:5000/get_graph_data', {  // Ensure the URL matches your Flask server's URL
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ scholarName: userInput })
  })
  .then(response => response.json())
  .then(data => {
    console.log('Graph Data:', data);
    // Process and visualize the graph data
    visualizeGraph(data);
  })
  .catch(error => {
    console.error('There was a problem with the fetch operation:', error);
  });
});

function visualizeGraph(data) {
  const svg = d3.select("#graph")
    .attr("width", 960)
    .attr("height", 600);

  // Clear previous graph
  svg.selectAll("*").remove();

  const color = d3.scaleOrdinal(d3.schemeCategory10);

  const simulation = d3.forceSimulation(data.nodes)
    .force("link", d3.forceLink(data.links).id(d => d.id))
    .force("charge", d3.forceManyBody())
    .force("center", d3.forceCenter(960 / 2, 600 / 2));

  const link = svg.append("g")
    .attr("class", "links")
    .selectAll("line")
    .data(data.links)
    .join("line")
      .attr("stroke-width", 1.5)
      .attr("stroke", "#999");

  const node = svg.append("g")
    .attr("class", "nodes")
    .selectAll("circle")
    .data(data.nodes)
    .join("circle")
      .attr("r", 5)
      .attr("fill", d => color(d.group))  // Color nodes based on group
      .call(drag(simulation))
      .on("mouseover", function(event, d) {
        d3.select(this).attr("r", 12);  // Increase node size on hover
      })
      .on("mouseout", function(event, d) {
        d3.select(this).attr("r", 5);  // Reset node size
      })
      .on("click", (event, d) => displayNodeInfo(d));

  simulation.on("tick", () => {
    link
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y);

    node
      .attr("cx", d => d.x)
      .attr("cy", d => d.y);
  });
}

function drag(simulation) {
  return d3.drag()
    .on("start", event => {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    })
    .on("drag", event => {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    })
    .on("end", event => {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    });
}

function displayNodeInfo(d) {
  const infoElement = document.getElementById('info');
  if (infoElement) {
    if (d.group === 1) {
      infoElement.innerHTML = `Institution: ${d.id}`;
    } 
    else {
    infoElement.innerHTML = `Scholar: ${d.id}`;
    }   } 
  else {
    console.error('Element with ID "info" not found.');
  }

}