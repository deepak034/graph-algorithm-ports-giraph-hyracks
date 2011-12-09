package org.apache.giraph.benchmark;

import java.io.IOException;
import java.util.Iterator;
import java.util.Map;
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
import org.apache.giraph.lib.TextVertexInputFormat.TextVertexReader;
import org.apache.giraph.lib.TextVertexOutputFormat;
import org.apache.giraph.lib.TextVertexOutputFormat.TextVertexWriter;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import com.google.common.collect.Maps;

public class PageRankVertex extends
		Vertex<LongWritable, DoubleWritable, FloatWritable, DoubleWritable>
		implements Tool {
	/** Configuration from Configurable */
	private Configuration conf;

	/** How many supersteps to run */
	public static String SUPERSTEP_COUNT = "PageRankVertex.superstepCount";

	@Override
	public void compute(Iterator<DoubleWritable> msgIterator) {
		if (getSuperstep() >= 1) {
			double sum = 0;
			while (msgIterator.hasNext()) {
				sum += msgIterator.next().get();
			}
			DoubleWritable vertexValue = new DoubleWritable(
					(0.15f / getNumVertices()) + 0.85f * sum);
			setVertexValue(vertexValue);
		}

		if (getSuperstep() < getConf().getInt(SUPERSTEP_COUNT, -1)) {
			long edges = getNumOutEdges();
			sendMsgToAllEdges(new DoubleWritable(getVertexValue().get() / edges));
		} else {
			voteToHalt();
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
		options.addOption("s", "supersteps", true,
				"Supersteps to execute before finishing");
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
		if (!cmd.hasOption('s')) {
			System.out.println("Need to set the number of supesteps (-s)");
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
		job.setVertexInputFormatClass(PageRankVertexInputFormat.class);
		job.setVertexOutputFormatClass(PageRankVertexOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(cmd.getOptionValue('i')));
		FileOutputFormat.setOutputPath(job, new Path(cmd.getOptionValue('o')));
		job.setWorkerConfiguration(workers, workers, 100.0f);
		job.getConfiguration().setInt(SUPERSTEP_COUNT,
				Integer.parseInt(cmd.getOptionValue('s')));

		if (cmd.hasOption('s')) {
			getConf().setInt(SUPERSTEP_COUNT,
					Integer.parseInt(cmd.getOptionValue('s')));
		}
		if (job.run(true) == true) {
			return 0;
		} else {
			return -1;
		}
	}

	public static void main(String[] args) throws Exception {
		System.exit(ToolRunner.run(new PageRankVertex(), args));
	}

	public static class PageRankVertexReader
			extends
			TextVertexReader<LongWritable, DoubleWritable, FloatWritable, DoubleWritable> {

		public PageRankVertexReader(
				RecordReader<LongWritable, Text> lineRecordReader) {
			super(lineRecordReader);
		}

		@Override
		public boolean nextVertex() throws IOException, InterruptedException {
			return getRecordReader().nextKeyValue();
		}

		@Override
		public BasicVertex<LongWritable, DoubleWritable, FloatWritable, DoubleWritable> getCurrentVertex()
				throws IOException, InterruptedException {

			BasicVertex<LongWritable, DoubleWritable, FloatWritable, DoubleWritable> vertex = BspUtils
					.<LongWritable, DoubleWritable, FloatWritable, DoubleWritable> createVertex(getContext()
							.getConfiguration());

			Text line = getRecordReader().getCurrentValue();
			try {
				StringTokenizer tokenizer = new StringTokenizer(line.toString());

				LongWritable vertexId = new LongWritable(
						Long.parseLong(tokenizer.nextToken()));
				
				float edgeValue = 0f;
				Map<LongWritable, FloatWritable> edges = Maps.newHashMap();
				while (tokenizer.hasMoreTokens()) {
					edges.put(
							new LongWritable(Long.parseLong(tokenizer
									.nextToken())),
							new FloatWritable(edgeValue));
				}
				DoubleWritable vertexValue = new DoubleWritable(1d / edges.size());
				vertex.initialize(vertexId, vertexValue, edges, null);
			} catch (Exception e) {
				throw new IllegalArgumentException(
						"next: Couldn't get vertex from line " + line, e);
			}

			return vertex;
		}

	}

	public static class PageRankVertexInputFormat
			extends
			TextVertexInputFormat<LongWritable, DoubleWritable, FloatWritable, DoubleWritable> {

		@Override
		public VertexReader<LongWritable, DoubleWritable, FloatWritable, DoubleWritable> createVertexReader(
				InputSplit split, TaskAttemptContext context)
				throws IOException {
			return new PageRankVertexReader(textInputFormat.createRecordReader(
					split, context));
		}

	}

	public static class PageRankVertexWriter extends
			TextVertexWriter<LongWritable, DoubleWritable, FloatWritable> {
		public PageRankVertexWriter(RecordWriter<Text, Text> lineRecordWriter) {
			super(lineRecordWriter);
		}

		@Override
		public void writeVertex(
				BasicVertex<LongWritable, DoubleWritable, FloatWritable, ?> vertex)
				throws IOException, InterruptedException {
			getRecordWriter().write(new Text(vertex.getVertexId().toString()),
					new Text(vertex.getVertexValue().toString()));
		}
	}

	public static class PageRankVertexOutputFormat extends
			TextVertexOutputFormat<LongWritable, DoubleWritable, FloatWritable> {

		@Override
		public VertexWriter<LongWritable, DoubleWritable, FloatWritable> createVertexWriter(
				TaskAttemptContext context) throws IOException,
				InterruptedException {
			RecordWriter<Text, Text> recordWriter = textOutputFormat
					.getRecordWriter(context);
			return new PageRankVertexWriter(recordWriter);
		}
	}
}
