const canvas = document.querySelector("canvas");

const size = Math.max(window.innerWidth, window.innerHeight);
canvas.width = size;
canvas.height = size;


const GRID_SIZE_X = Math.floor(canvas.width / 16);
const GRID_SIZE_Y = Math.floor(canvas.width / 16);
const UPDATE_INTERVAL = 200; // Update every 200ms (5 times/sec)
let step = 0; // Track how many simulation steps have been run
const WORKGROUP_SIZE = 8;

// Your WebGPU code will begin here!
if (!navigator.gpu) {
    // add warning to page
    document.body.className = "error";
    document.body.innerHTML = "Your browser does not support WebGPU! </br><a href='https://caniuse.com/webgpu'>supported browsers</a>";
    document.body.style.color = "red";
    document.body.style.fontSize = "x-large";
    document.body.style.fontWeight = "bold";
    document.body.style.textAlign = "center";
    document.body.style.paddingTop = "45vh";
    document.body.style.backgroundColor = "black";
    document.body.style.margin = "0";
    document.body.style.height = "100vh";
    document.body.style.width = "100vw";
    document.body.style.overflow = "hidden";
    document.body.style.fontFamily = "roboto, sans-serif";
    throw new Error("WebGPU not supported on this browser.");
}
const adapter = await navigator.gpu.requestAdapter({powerPreference: 'low-power'});
if (!adapter) {
    throw new Error("No appropriate GPUAdapter found.");
}
const device = await adapter.requestDevice();

const context = canvas.getContext("webgpu");
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
    device: device,
    format: canvasFormat
});


// Create a buffer that describes the vertices of a cell.
const vertices = new Float32Array([
    //   X,    Y,
    -0.8, -0.8, // Triangle 1 (Blue)
    0.8, -0.8,
    0.8,  0.8,

    -0.8, -0.8, // Triangle 2 (Red)
    0.8,  0.8,
    -0.8,  0.8,
]);
const vertexBuffer = device.createBuffer({
    label: "Cell vertices",
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
})
device.queue.writeBuffer(vertexBuffer, /*bufferOffset=*/0, vertices);
const vertexBufferLayout = {
    arrayStride: 8,
    attributes: [{
        format: "float32x2",
        offset: 0,
        shaderLocation: 0, // Position, see vertex shader
    }],
};

// Create a uniform buffer that describes the grid.
const uniformArray = new Float32Array([GRID_SIZE_X, GRID_SIZE_Y]);

const uniformBuffer = device.createBuffer({
    label: "Grid Uniforms",
    size: uniformArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

// Create an array representing the active state of each cell.
const cellStateArray = new Uint32Array(GRID_SIZE_X * GRID_SIZE_Y);

// Create a storage buffer to hold the cell state.
const cellStateStorage = [
    device.createBuffer({
        label: "Cell State A",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    device.createBuffer({
        label: "Cell State B",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
];

// Mark every third cell of the grid as active.
for (let i = 0; i < cellStateArray.length; ++i) {
    cellStateArray[i] = Math.random() > 0.6 ? 1 : 0;
    // cellStateArray[i] = 0;
}
device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);


// Create a shader module describing both the vertex and fragment shaders.
const cellShaderModule = device.createShaderModule({
    label: "Cell shader",
    code: /* wgsl */`
        struct VertexInput {
            @location(0) pos: vec2f,
            @builtin(instance_index) instance: u32,
        };
            
        struct VertexOutput {
            @builtin(position) pos: vec4f,
            @location(0) cell: vec2f,
        };

        @group(0) @binding(0) var<uniform> grid: vec2f;
        @group(0) @binding(1) var<storage> cellState: array<u32>;

        @vertex
        fn vertexMain(input: VertexInput) ->
        VertexOutput{
            let i = f32(input.instance);
            let cell = vec2f(i % grid.x, floor(i / grid.y));
            let state = f32(cellState[input.instance]);

            let cellOffset = cell / grid * 2;
            let gridPos = (input.pos * state + 1) / grid - 1 + cellOffset;

            var output: VertexOutput;
            output.pos = vec4f(gridPos, 0, 1);
            output.cell = cell;
            return output;
        }

        @fragment
        fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
            let c = input.cell / grid;
            return vec4f(0, 0.6, 0, 1); // (Red, Green, Blue, Alpha)
        }
    `
});

// Create the compute shader that will process the simulation.
const simulationShaderModule = device.createShaderModule({
    label: "Game of Life simulation shader",
    code: /* wgsl */`
        @group(0) @binding(0) var<uniform> grid: vec2f;
        @group(0) @binding(1) var<storage> cellStateIn: array<u32>;
        @group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;\

        fn cellIndex(cell: vec2u) -> u32 {
            return (cell.y % u32(grid.y)) * u32(grid.x) +
                (cell.x % u32(grid.x));
        }

        fn cellActive(x: u32, y: u32) -> u32 {
            return cellStateIn[cellIndex(vec2(x, y))];
        }

        @compute
        @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
        fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
            // Determine how many active neighbors this cell has.
            let activeNeighbors = cellActive(cell.x+1, cell.y+1) +
                cellActive(cell.x+1, cell.y) +
                cellActive(cell.x+1, cell.y-1) +
                cellActive(cell.x, cell.y-1) +
                cellActive(cell.x-1, cell.y-1) +
                cellActive(cell.x-1, cell.y) +
                cellActive(cell.x-1, cell.y+1) +
                cellActive(cell.x, cell.y+1);

            let i = cellIndex(cell.xy);

            // Conway's game of life rules:
            switch activeNeighbors {
                case 2: { // Active cells with 2 neighbors stay active.
                    cellStateOut[i] = cellStateIn[i];
                }
                case 3: { // Cells with 3 neighbors become or stay active.
                    cellStateOut[i] = 1;
                }
                default: { // Cells with < 2 or > 3 neighbors become inactive.
                    cellStateOut[i] = 0;
                }
            }
            
    }`
  });

const bindGroupLayout = device.createBindGroupLayout({
    label: "Cell Bind Group Layout",
    entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT  ,
        buffer: {}
    }, {
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage"} // Cell state input buffer
    }, {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage"} // Cell state output buffer
    }]
});

// Create a bind group that associates the uniform buffer with the uniform
const bindGroups = [
    device.createBindGroup({
        label: "Cell renderer bind group A",
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: { buffer: uniformBuffer }
            }, {
                binding: 1,
                resource: { buffer: cellStateStorage[0] }
            }, {
                binding: 2,
                resource: { buffer: cellStateStorage[1] }
            }
        ],
    }),
    device.createBindGroup({
        label: "Cell renderer bind group B",
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: { buffer: uniformBuffer }
            }, {
                binding: 1,
                resource: { buffer: cellStateStorage[1] }
            }, {
                binding: 2,
                resource: { buffer: cellStateStorage[0] }
            }
        ],
    })
];

const pipelineLayout = device.createPipelineLayout({
    label: "Cell Pipeline Layout",
    bindGroupLayouts: [ bindGroupLayout ],
});


// Create a render pipeline that describes the entire graphics pipeline.
const cellPipeline = device.createRenderPipeline({
    label: "Cell pipeline",
    layout: pipelineLayout,
    vertex: {
        module: cellShaderModule,
        entryPoint: "vertexMain",
        buffers: [vertexBufferLayout],
    },
    fragment : {
        module: cellShaderModule,
        entryPoint: "fragmentMain",
        targets: [{
            format: canvasFormat
        }]
    }
});

// Create a compute pipeline that updates the game state.
const simulationPipeline = device.createComputePipeline({
    label: "Simulation pipeline",
    layout: pipelineLayout,
    compute: {
      module: simulationShaderModule,
      entryPoint: "computeMain",
    }
});



function updateGrid() {
    // Create a command encoder and pass in the render pipeline.
    const encoder =  device.createCommandEncoder();

    // Run the simulation.
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(simulationPipeline),
    computePass.setBindGroup(0, bindGroups[step % 2]);

    const workgroupCount = Math.ceil((GRID_SIZE_X + GRID_SIZE_Y) / 2 / WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(workgroupCount, workgroupCount);
    computePass.end();
    
    
    step++;

    // Render the grid.
    const pass = encoder.beginRenderPass({
        colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadOp: "clear",
            clearValue: { r: 0.01, g: 0.01, b: 0.01, a: 1.0 },
            storeOp: "store",
        }]
    });
    pass.setPipeline(cellPipeline);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.setBindGroup(0, bindGroups[step % 2]);
    pass.draw(vertices.length / 2, GRID_SIZE_X * GRID_SIZE_Y);
    pass.end();

    const commandBuffer = encoder.finish();

    device.queue.submit([commandBuffer]);
}


setInterval(updateGrid, UPDATE_INTERVAL);


