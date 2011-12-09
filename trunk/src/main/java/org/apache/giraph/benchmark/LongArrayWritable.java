package org.apache.giraph.benchmark;

import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.LongWritable;

public class LongArrayWritable extends ArrayWritable {
	public LongArrayWritable() {
		super(LongWritable.class);
	}

	public LongArrayWritable(Class<LongWritable> valueClass, LongWritable[] value) {
		super(valueClass, value);
	}
}
