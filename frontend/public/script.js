console.log("Script loaded");

function fetchData() {
  return fetch('data/graph.json')
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return response.json();
    });
}

fetchData().then(data => {
  console.log("Data loaded:", data);

  const svg = d3.select("#graph")
    .attr("width", 960)
    .attr("height", 600);

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
      .attr("fill", "red")
      .call(drag(simulation));

  const label = svg.append("g")
    .attr("class", "labels")
    .selectAll("text")
    .data(data.nodes)
    .join("text")
      .attr("text-anchor", "middle")
      .attr("dy", -10)
      .text(d => d.id);

  simulation.on("tick", () => {
    link
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y);

    node
      .attr("cx", d => d.x)
      .attr("cy", d => d.y);

    label
      .attr("x", d => d.x)
      .attr("y", d => d.y);
  });

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
}).catch(error => {
  console.error("Error loading data:", error);
});