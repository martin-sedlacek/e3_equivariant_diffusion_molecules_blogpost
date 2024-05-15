### GRAM Workshop

The Geometric-Grounded Representation Learning and Generative Modelling [GRaM](https://gram-workshop.github.io/) workshop aims to provide a platform that fosters learning, collaboration, and the advancement of the geometry-grounded methods in machine learning. It includes three tracks: a proceedings track used to publish original research, a tutorial track in the form of a Google Collab, and a blogpost track. In this track, we intend to encourage transparent discussions and opinions in the field, and make the science more accessible.

### Call for blog posts

The format and process for this blog post track is as follows:

- The post can be written on any subject linked to the [GRaM workshop](https://gram-workshop.github.io/). It can offer insights on certain aspects of the field, list a series of open problems, explain mathematical concepts in a pedagogical way, or discuss a research paper.

- The blogs will be peer-reviewed in a double-blind way. To be accepted, submissions must meet criteria such as content quality and novelty, clear and pedagogical presentation, new insights in theory or practice, and reproducibility or enhancement of experiments.

- The posts will be hosted on this website.

Note that the submission, reviewing and publication process of the blogposts are directly inspired by the [ICLR track](https://iclr-blogposts.github.io/). See the [instruction page]({{ '/instructions/'}}) for detailed instructions.

### Instructions for Submitting a Blog Post

1. **Fork and Rename**:

   - Navigate to the [staging repository](https://github.com/gram-blogposts/staging/).
   - Click on the "Fork" button of the repository page to create a copy of the repository under your GitHub account.
   - After forking, rename your repository to reflect the topic of your submission.

2. **Setup and Deployment**:

   - Clone your forked repository to your local machine.
   - Setup locally the webiste.
     Assuming you have [Ruby](https://www.ruby-lang.org/en/downloads/) and [Bundler](https://bundler.io/) installed on your system, try:

   ```bash
   $ git clone git@github.com:<your-username>/<your-repo-name>.git
   $ cd <your-repo-name>
   $ bundle install
   $ bundle exec jekyll serve
   ```

   - Your can locally access and modify the website on the local address: `http://127.0.0.1:4000/`.

3. **Preparing Your Submission**:
   You can copy and modify the [Distill Template]({% post_url 2018-12-22-distill %}), found under the `_posts` directory. Leave the original template as it is.

   - Create a Markdown or HTML file in the `_posts/` directory with the format `_posts/YYYY-MM-DD-your-submission.md`.
   - Add any static image to `assets/img/YYYY-MM-DD-your-submission/`.
   - Add any interactive plotly figures to `assets/plotly/YYYY-MM-DD-your-submission/`.
   - Put your citations into a bibtex file in `assets/bibliography/YYYY-MM-DD-your-submission.bib`.

4. **Submit a Pull Request**:

   - Before submitting, double-check that all personal identifiers have been removed from your blog post content.
   - Push your changes to your forked repository.
   - Go to your forked repository on GitHub and click on "New Pull Request".
   - Ensure the pull request is directed to the staging repository and name the pull request after your submission topic.
   - Provide a clear and concise description of your changes in the pull request description.

5. **Submit to Open Review**:
   - Once your pull request is submitted, provide the title and URL of your blog post via OpenReview.

### Review and Publication

- **Review**: Reviewers will assess only the live content of the blog. They are expected to avoid probing into the repository's historical data to maintain anonymity.
- **Camera-Ready Submission**: Upon acceptance, add your identifying information to the post. Submit a final pull request to the main repository for integration into the official blog. Track chairs may request additional edits before final approval.

### Further questions

- Our Blogpost track is directly inspired by the ICLR Blogpost track. If you have problems building your website or modifying any file, you can have a look at their [instructions](https://iclr-blogposts.github.io/2024/submitting/).
- Feel free to reach out to the organizers at: [organizers AT gram-workshop DOT org](mailto:organizers@gram-workshop.org)
