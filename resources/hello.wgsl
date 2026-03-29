@group(0) @binding(0) var<storage,read> inputBuffer: array<f32, 64>;
@group(0) @binding(1) var<storage,read_write> outputBuffer: array<f32, 64>;

fn f(i: f32) -> f32{
	return i * 2.0;
}

@compute @workgroup_size(32) 
fn computeStuff(@builtin(global_invocation_id) id: vec3<u32>)  { 
	outputBuffer[id.x] = inputBuffer[id.x];
}

