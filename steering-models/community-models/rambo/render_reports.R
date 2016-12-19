render_report = function(submission_filename, output_filename){
  rmarkdown::render("evaluation_report.Rmd", params = list(
    input_data = submission_filename
  ),
  output_file = output_filename,
  output_dir = "reports")
}


render_report("submissions/submission_g_d2_n1_n2_co_e_phase2.csv",
              "temp_report.html")