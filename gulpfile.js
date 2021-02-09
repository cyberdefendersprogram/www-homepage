const gulp = require('gulp');

gulp.task('css', () => {
    console.log("Doing the CSS thing");
});

gulp.task('default', gulp.series('css'));