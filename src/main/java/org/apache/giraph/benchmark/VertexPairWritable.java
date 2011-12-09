package org.apache.giraph.benchmark;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Writable;

public class VertexPairWritable implements Writable {
	
	private LongWritable middleVertex;
	private LongArrayWritable firstVertexList;

	public VertexPairWritable() {
		set(new LongWritable(), new LongArrayWritable());
	}
	
	public VertexPairWritable(LongWritable middle, LongArrayWritable list) {
		set(middle, list);
	}
	
	public void set(LongWritable middle, LongArrayWritable list) {
		middleVertex = middle;
		firstVertexList = list;
	}
	
	public LongWritable getMiddleVertex() {
		return middleVertex;
	}
	
	public LongArrayWritable getFirstVertexList() {
		return firstVertexList;
	}
	
	@Override
	public void readFields(DataInput in) throws IOException {
		middleVertex.readFields(in);
		firstVertexList.readFields(in);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		middleVertex.write(out);
		firstVertexList.write(out);
	}

}
