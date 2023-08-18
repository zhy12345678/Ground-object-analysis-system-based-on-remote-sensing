package mine.design.en.biz;

import mine.design.en.entity.Picture;

import java.util.List;

public interface PictureBiz {
    void add(Picture picture);
    List<Picture> getTime();
    List<Picture> getAccurate();
    List<Picture> getAll();



}
