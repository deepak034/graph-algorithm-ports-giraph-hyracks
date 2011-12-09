package org.apache.giraph.benchmark;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.StringTokenizer;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.PosixParser;
import org.apache.giraph.graph.BasicVertex;
import org.apache.giraph.graph.BspUtils;
import org.apache.giraph.graph.GiraphJob;
import org.apache.giraph.graph.Vertex;
import org.apache.giraph.graph.VertexReader;
import org.apache.giraph.graph.VertexWriter;
import org.apache.giraph.lib.TextVertexInputFormat;
import org.apache.giraph.lib.TextVertexOutputFormat;
import org.apache.giraph.lib.TextVertexInputFormat.TextVertexReader;
import org.apache.giraph.lib.TextVertexOutputFormat.TextVertexWriter;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import com.google.common.collect.Maps;

/*
 * Type Parameters:
 * I - vertex id
 * V - vertex data
 * E - edge data
 * M - message data
 */
public class BipartiteVertex extends
		Vertex<LongWritable, ValueWritable, FloatWritable, LongWritable>
		implements Tool {

	/** Configuration from Configurable */
	private Configuration conf;

	@Override
	public void compute(Iterator<LongWritable> msgIterator) throws IOException {

		if (getSuperstep() % 4 == 0) {
			if (getVertexValue().getVertexType() == ValueWritable.LEFT_VERTEX) {
				sendMsgToAllEdges(getVertexId());
				voteToHalt();
			}
		} else if (getSuperstep() % 4 == 1) {
			if (getVertexValue().getVertexType() == ValueWritable.RIGHT_VERTEX) {
				// randomly select one of the incoming messages and send the
				// vertex id as the message to the source of the selected
				// message
				List<LongWritable> msgList = getMsgList();
				Random rand = new Random();
				int randIndex = rand.nextInt(msgList.size());
				sendMsg(msgList.get(randIndex), getVertexId());
				voteToHalt();
			}
		} else if (getSuperstep() % 4 == 2) {
			if (getVertexValue().getVertexType() == ValueWritable.LEFT_VERTEX) {
				// randomly select one of the incoming messages, change the
				// matched id to the source of the selected message and send an
				// ack
				List<LongWritable> msgList = getMsgList();
				Random rand = new Random();
				int randIndex = rand.nextInt(msgList.size());
				ValueWritable value = getVertexValue();
				value.setMatchedVertexId(msgList.get(randIndex).get());
				setVertexValue(value);
				sendMsg(msgList.get(randIndex), getVertexId());
				voteToHalt();
			}
		} else if (getSuperstep() % 4 == 3) {
			if (getVertexValue().getVertexType() == ValueWritable.RIGHT_VERTEX) {
				// get the ack and change the matched id to the source of ack
				if (msgIterator.hasNext()) {
					ValueWritable value = getVertexValue();
					value.setMatchedVertexId(msgIterator.next().get());
					setVertexValue(value);
				}
				voteToHalt();
			}
		}
	}

	@Override
	public Configuration getConf() {
		return conf;
	}

	@Override
	public void setConf(Configuration conf) {
		this.conf = conf;
	}

	@Override
	public int run(String[] args) throws Exception {
		Options options = new Options();
		options.addOption("h", "help", false, "Help");
		options.addOption("w", "workers", true, "Number of workers");
		options.addOption("i", "input", true, "Input file");
		options.addOption("o", "output", true, "Output file");

		HelpFormatter formatter = new HelpFormatter();
		if (args.length == 0) {
			formatter.printHelp(getClass().getName(), options, true);
			return 0;
		}
		CommandLineParser parser = new PosixParser();
		CommandLine cmd = parser.parse(options, args);
		if (cmd.hasOption('h')) {
			formatter.printHelp(getClass().getName(), options, true);
			return 0;
		}
		if (!cmd.hasOption('w')) {
			System.out.println("Need to choose the number of workers (-w)");
			return -1;
		}
		if (!cmd.hasOption('i')) {
			System.out.println("Need to set the input file (-i)");
			return -1;
		}
		if (!cmd.hasOption('o')) {
			System.out.println("Need to set the output file (-o)");
			return -1;
		}

		int workers = Integer.parseInt(cmd.getOptionValue('w'));
		GiraphJob job = new GiraphJob(getConf(), getClass().getName());
		job.setVertexClass(getClass());
		job.setVertexInputFormatClass(BipartiteVertexInputFormat.class);
		job.setVertexOutputFormatClass(BipartiteVertexOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(cmd.getOptionValue('i')));
		FileOutputFormat.setOutputPath(job, new Path(cmd.getOptionValue('o')));
		job.setWorkerConfiguration(workers, workers, 100.0f);

		if (job.run(true) == true) {
			return 0;
		} else {
			return -1;
		}
	}

	public static void main(String[] args) throws Exception {
		System.exit(ToolRunner.run(new BipartiteVertex(), args));
	}

	public static class BipartiteVertexReader extends
			TextVertexReader<LongWritable, ValueWritable, FloatWritable, LongWritable> {

		public BipartiteVertexReader(
				RecordReader<LongWritable, Text> lineRecordReader) {
			super(lineRecordReader);
		}
		
		@Override
		public boolean nextVertex() throws IOException, InterruptedException {
			return getRecordReader().nextKeyValue();
		}

		
		@Override
		public BasicVertex<LongWritable, ValueWritable, FloatWritable, LongWritable> getCurrentVertex()
				throws IOException, InterruptedException {

			BasicVertex<LongWritable, ValueWritable, FloatWritable, LongWritable> vertex = BspUtils
					.<LongWritable, ValueWritable, FloatWritable, LongWritable> createVertex(getContext()
							.getConfiguration());

			Text line = getRecordReader().getCurrentValue();
			try {
				StringTokenizer tokenizer = new StringTokenizer(line.toString());

				LongWritable vertexId = new LongWritable(
						Long.parseLong(tokenizer.nextToken()));
				
				String type = tokenizer.nextToken();
				int vertexType = ValueWritable.LEFT_VERTEX;
				if (type == "R") {
					vertexType = ValueWritable.RIGHT_VERTEX;
				}
				ValueWritable vertexValue = new ValueWritable();
				vertexValue.set(vertexType, vertexId.get());
				
				float edgeValue = 0f;
				Map<LongWritable, FloatWritable> edges = Maps.newHashMap();
				while (tokenizer.hasMoreTokens()) {
					edges.put(
							new LongWritable(Long.parseLong(tokenizer
									.nextToken())),
							new FloatWritable(edgeValue));
				}
				
				vertex.initialize(vertexId, vertexValue, edges, null);
			} catch (Exception e) {
				throw new IllegalArgumentException(
						"next: Couldn't get vertex from line " + line, e);
			}

			return vertex;
		}

	}

	public static class BipartiteVertexInputFormat extends
			TextVertexInputFormat<LongWritable, ValueWritable, FloatWritable, LongWritable> {

		@Override
		public VertexReader<LongWritable, ValueWritable, FloatWritable, LongWritable> createVertexReader(
				InputSplit split, TaskAttemptContext context)
				throws IOException {
			return new BipartiteVertexReader(
					textInputFormat.createRecordReader(split, context));
		}

	}

	public static class BipartiteVertexWriter extends
			TextVertexWriter<LongWritable, ValueWritable, FloatWritable> {
		public BipartiteVertexWriter(RecordWriter<Text, Text> lineRecordWriter) {
			super(lineRecordWriter);
		}

		@Override
		public void writeVertex(
				BasicVertex<LongWritable, ValueWritable, FloatWritable, ?> vertex)
				throws IOException, InterruptedException {
			if (vertex.getVertexValue().getVertexType() == ValueWritable.LEFT_VERTEX
					&& vertex.getVertexValue().getMatchedVertexId() > 0) {
				// write the matching pair
				getRecordWriter()
						.write(new Text(vertex.getVertexId() + "\t"
								+ vertex.getVertexValue().getMatchedVertexId()),
								null);
			}
		}
	}

	public static class BipartiteVertexOutputFormat extends
			TextVertexOutputFormat<LongWritable, ValueWritable, FloatWritable> {

		@Override
		public VertexWriter<LongWritable, ValueWritable, FloatWritable> createVertexWriter(
				TaskAttemptContext context) throws IOException,
				InterruptedException {
			RecordWriter<Text, Text> recordWriter = textOutputFormat
					.getRecordWriter(context);
			return new BipartiteVertexWriter(recordWriter);
		}
	}

}
