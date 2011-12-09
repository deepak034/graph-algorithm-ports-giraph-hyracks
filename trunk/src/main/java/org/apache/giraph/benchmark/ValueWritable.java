package org.apache.giraph.benchmark;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

public class ValueWritable implements Writable {    
	public static final int LEFT_VERTEX = 0;
	public static final int RIGHT_VERTEX = 1;
	
    private int vertexType;
    private long matchedVertexId;
            
    public void set(int type, long id) {
    	vertexType = type;
    	matchedVertexId = id;
    }
    
    public int getVertexType() {
    	return vertexType;
    }
    public long getMatchedVertexId() {
    	return matchedVertexId;
    }
    public void setMatchedVertexId(long id) {
    	matchedVertexId = id;
    }

    @Override
    public void write(DataOutput out) throws IOException {
    	out.writeInt(vertexType);
    	out.writeLong(matchedVertexId);
    }
    
    @Override
    public void readFields(DataInput in) throws IOException {
    	vertexType = in.readInt();
    	matchedVertexId = in.readLong();
    }
  }